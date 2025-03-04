"""
Data augmentation techniques for sinogram data (DISABLED)

This module contains augmentation techniques that are currently disabled
but kept for reference. The code is not used in the current implementation.
"""
import torch
import random
import numpy as np

class SinogramAugmentation:
    """
    Augmentation pipeline for sinogram data. (DISABLED)
    
    Note: This class is kept for reference but is not used in the current implementation
    since data augmentation is disabled.
    """
    def __init__(self, rotate_prob=0.0, noise_prob=0.0, mask_prob=0.0, noise_level=0.0):
        self.rotate_prob = rotate_prob
        self.noise_prob = noise_prob
        self.mask_prob = mask_prob
        self.noise_level = noise_level
    
    def __call__(self, sinogram):
        """
        This method is not used as augmentation is disabled.
        Returns the input sinogram without modification.
        """
        return sinogram.clone()

class SinogramMixup:
    """
    Mixup augmentation for sinogram data. (DISABLED)
    
    Note: This class is kept for reference but is not used in the current implementation
    since data augmentation is disabled.
    """
    def __init__(self, alpha=0.0):
        self.alpha = alpha
    
    def __call__(self, batch):
        """
        This method is not used as augmentation is disabled.
        Returns the input batch without modification.
        """
        return batch

class RandomAngularMasking:
    """
    Randomly mask out angular sections of the sinogram. (DISABLED)
    
    Note: This class is kept for reference but is not used in the current implementation
    since data augmentation is disabled.
    """
    def __init__(self, min_angle=0, max_angle=0, num_sections=0):
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.num_sections = num_sections
    
    def __call__(self, sinogram):
        """
        This method is not used as augmentation is disabled.
        Returns the input sinogram without modification.
        """
        return sinogram.clone()