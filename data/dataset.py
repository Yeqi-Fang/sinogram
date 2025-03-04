"""
Dataset implementation for sinogram data with incomplete ring
"""
import torch
from torch.utils.data import Dataset
import random
import numpy as np

class SinogramDataset(Dataset):
    """
    Dataset for sinogram data with incomplete ring reconstruction.
    
    Args:
        sinogram_path (str): Path to the complete sinogram data
        mask_angles (tuple): Range of angles to mask (e.g., (30, 60) for 30-60 degrees)
        block_size (int): Size of each block along the ring difference dimension
    """
    def __init__(self, sinogram_path, mask_angles=None, transform=None, 
                 mask_paths=None, use_predefined_masks=False, block_size=64):
        # Note: transform, mask_paths, and use_predefined_masks parameters are kept
        # for backward compatibility but are not used since augmentation is disabled
        self.sinogram_data = torch.load(sinogram_path)
        self.transform = transform
        self.mask_angles = mask_angles
        self.mask_paths = mask_paths
        self.use_predefined_masks = use_predefined_masks
        self.block_size = block_size
        
        if use_predefined_masks and mask_paths:
            self.masks = [torch.load(path) for path in mask_paths]
        else:
            self.masks = None
        
        # Calculate total number of blocks
        self.total_blocks = self.sinogram_data.shape[2] // self.block_size
        if self.sinogram_data.shape[2] % self.block_size != 0:
            self.total_blocks += 1  # Handle partial last block
        
    def __len__(self):
        return self.total_blocks
        
    def __getitem__(self, idx):
        # Extract a block of the sinogram
        start_idx = idx * self.block_size
        end_idx = min(start_idx + self.block_size, self.sinogram_data.shape[2])
        
        # Handle case where the last block might be smaller
        if end_idx - start_idx < self.block_size:
            # Pad the last block to ensure consistent size
            block = torch.zeros(self.sinogram_data.shape[0], 
                               self.sinogram_data.shape[1], 
                               self.block_size, 
                               dtype=self.sinogram_data.dtype)
            block[:, :, :(end_idx - start_idx)] = self.sinogram_data[:, :, start_idx:end_idx]
        else:
            block = self.sinogram_data[:, :, start_idx:end_idx].clone()
        
        # Create ground truth (complete sinogram block)
        target = block.clone()
        
        # Apply mask to create incomplete sinogram
        if self.use_predefined_masks and self.masks:
            # Use a random predefined mask
            mask = random.choice(self.masks)
            block = block * mask[:, :, :block.shape[2]]  # Apply mask to block
        elif self.mask_angles is not None:
            # Convert angles to indices in the sinogram
            angle_range = 180  # Assuming the sinogram covers 180 degrees
            total_angles = self.sinogram_data.shape[0]
            start_angle_idx = int(self.mask_angles[0] / angle_range * total_angles)
            end_angle_idx = int(self.mask_angles[1] / angle_range * total_angles)
            
            # Mask the specified angular range
            block[start_angle_idx:end_angle_idx, :, :] = 0
        
        # Apply transformations if needed
        if self.transform:
            block = self.transform(block)
        
        return block, target

class SinogramPatchDataset(Dataset):
    """
    Dataset that creates patches from the sinogram for more efficient training.
    
    Args:
        sinogram_path (str): Path to the complete sinogram data
        patch_size (tuple): Size of patches to extract (H, W, C)
        mask_angles (tuple): Range of angles to mask (e.g., (30, 60) for 30-60 degrees)
        transform (callable, optional): Optional transform to be applied to the data
        num_patches (int): Number of patches to extract
    """
    def __init__(self, sinogram_path, patch_size=(112, 112, 64), 
                 mask_angles=None, transform=None, num_patches=1000):
        self.sinogram_data = torch.load(sinogram_path)
        self.patch_size = patch_size
        self.mask_angles = mask_angles
        self.transform = transform
        self.num_patches = num_patches
        
        # Calculate valid patch extraction ranges
        self.h_range = self.sinogram_data.shape[0] - patch_size[0]
        self.w_range = self.sinogram_data.shape[1] - patch_size[1]
        self.c_range = self.sinogram_data.shape[2] - patch_size[2]
        
    def __len__(self):
        return self.num_patches
    
    def __getitem__(self, idx):
        # Randomly sample patch location
        h_start = random.randint(0, max(0, self.h_range))
        w_start = random.randint(0, max(0, self.w_range))
        c_start = random.randint(0, max(0, self.c_range))
        
        # Extract patch
        patch = self.sinogram_data[
            h_start:h_start+self.patch_size[0],
            w_start:w_start+self.patch_size[1],
            c_start:c_start+self.patch_size[2]
        ].clone()
        
        # Create ground truth (complete patch)
        target = patch.clone()
        
        # Apply mask to create incomplete sinogram patch
        if self.mask_angles is not None:
            # Convert angles to indices in the sinogram
            angle_range = 180  # Assuming the sinogram covers 180 degrees
            total_angles = self.sinogram_data.shape[0]
            start_idx = int(self.mask_angles[0] / angle_range * total_angles)
            end_idx = int(self.mask_angles[1] / angle_range * total_angles)
            
            # Adjust indices to patch coordinates
            start_rel = max(0, start_idx - h_start)
            end_rel = min(self.patch_size[0], end_idx - h_start)
            
            if start_rel < end_rel:  # Only apply if mask overlaps with patch
                patch[start_rel:end_rel, :, :] = 0
        
        # Apply transformations if needed
        if self.transform:
            patch = self.transform(patch)
            
        return patch, target