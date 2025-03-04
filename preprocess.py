"""
Preprocess sinogram data for the PET reconstruction model
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from data.utils import normalize_sinogram, create_angle_mask, apply_mask
from config import Config

def preprocess_sinogram(sinogram_path, output_path, apply_normalization=True):
    """
    Preprocess the sinogram data and save it.
    
    Args:
        sinogram_path (str): Path to the raw sinogram data
        output_path (str): Path to save the processed data
        apply_normalization (bool): Whether to normalize the data
    """
    print(f"Loading sinogram from {sinogram_path}")
    sinogram = torch.load(sinogram_path)
    
    print(f"Original sinogram shape: {sinogram.shape}")
    
    # Apply normalization if requested
    if apply_normalization:
        print("Normalizing sinogram data...")
        sinogram = normalize_sinogram(sinogram)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the processed sinogram
    print(f"Saving processed sinogram to {output_path}")
    torch.save(sinogram, output_path)
    
    # Generate and save a visualization of the first slice
    plt.figure(figsize=(10, 4))
    plt.imshow(sinogram[0, :, :64].T, cmap='magma', aspect='auto')
    plt.title("Processed Sinogram (First Slice)")
    plt.colorbar()
    plt.savefig(os.path.join(os.path.dirname(output_path), 'sinogram_visualization.png'))
    plt.close()
    
    print("Preprocessing completed!")

def create_masks(sinogram_shape, output_dir, num_masks=5):
    """
    Create and save various angular masks for data augmentation.
    
    Args:
        sinogram_shape (tuple): Shape of the sinogram
        output_dir (str): Directory to save masks
        num_masks (int): Number of different masks to create
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create masks with different angular ranges
    for i in range(num_masks):
        # Randomly select an angular range to mask
        start_angle = np.random.uniform(0, 150)
        end_angle = start_angle + np.random.uniform(10, 60)
        
        # Create the mask
        mask = create_angle_mask(sinogram_shape, start_angle, end_angle)
        
        # Save the mask
        torch.save(mask, os.path.join(output_dir, f'mask_{i+1}.pt'))
        
        # Generate and save a visualization
        plt.figure(figsize=(10, 4))
        plt.imshow(mask[:, :, 0].numpy(), cmap='gray', aspect='auto')
        plt.title(f"Mask {i+1}: {start_angle:.1f}° - {end_angle:.1f}°")
        plt.colorbar()
        plt.savefig(os.path.join(output_dir, f'mask_{i+1}_visualization.png'))
        plt.close()
    
    print(f"Created {num_masks} masks in {output_dir}")

def analyze_sinogram(sinogram_path):
    """
    Analyze the sinogram data and print statistics.
    
    Args:
        sinogram_path (str): Path to the sinogram data
    """
    print(f"Loading sinogram from {sinogram_path}")
    sinogram = torch.load(sinogram_path)
    
    print(f"Sinogram shape: {sinogram.shape}")
    print(f"Sinogram data type: {sinogram.dtype}")
    print(f"Sinogram min value: {sinogram.min()}")
    print(f"Sinogram max value: {sinogram.max()}")
    print(f"Sinogram mean value: {sinogram.mean()}")
    print(f"Sinogram standard deviation: {sinogram.std()}")
    
    # Check for NaN or Inf values
    nan_count = torch.isnan(sinogram).sum().item()
    inf_count = torch.isinf(sinogram).sum().item()
    print(f"Number of NaN values: {nan_count}")
    print(f"Number of Inf values: {inf_count}")
    
    # Check for zero values
    zero_count = (sinogram == 0).sum().item()
    zero_percentage = zero_count / sinogram.numel() * 100
    print(f"Number of zero values: {zero_count} ({zero_percentage:.2f}%)")
    
    # Visualize histogram
    plt.figure(figsize=(10, 6))
    plt.hist(sinogram.flatten().numpy(), bins=100)
    plt.title("Sinogram Value Distribution")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.savefig("sinogram_histogram.png")
    plt.close()
    
    # Visualize different slices
    num_slices = min(4, sinogram.shape[2])
    fig, axes = plt.subplots(1, num_slices, figsize=(20, 5))
    
    for i in range(num_slices):
        slice_idx = i * (sinogram.shape[2] // num_slices)
        axes[i].imshow(sinogram[:, :, slice_idx].numpy(), cmap='magma', aspect='auto')
        axes[i].set_title(f"Slice {slice_idx}")
        axes[i].set_xlabel("Radial Position")
        axes[i].set_ylabel("Angle")
    
    plt.tight_layout()
    plt.savefig("sinogram_slices.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Preprocess sinogram data for PET reconstruction")
    parser.add_argument("--sinogram_path", type=str, default=Config.SINOGRAM_PATH,
                        help="Path to the raw sinogram data")
    parser.add_argument("--output_path", type=str, default=Config.PROCESSED_SINOGRAM_PATH,
                        help="Path to save the processed data")
    parser.add_argument("--normalize", action="store_true", default=True,
                        help="Whether to normalize the data")
    parser.add_argument("--analyze", action="store_true",
                        help="Whether to analyze the sinogram data")
    parser.add_argument("--create_masks", action="store_true",
                        help="Whether to create masks for data augmentation")
    parser.add_argument("--num_masks", type=int, default=Config.NUM_MASKS,
                        help="Number of masks to create")
    
    args = parser.parse_args()
    
    # Analyze the sinogram if requested
    if args.analyze:
        analyze_sinogram(args.sinogram_path)
    
    # Preprocess the sinogram
    preprocess_sinogram(args.sinogram_path, args.output_path, args.normalize)
    
    # Create masks for data augmentation if requested
    if args.create_masks:
        create_masks(Config.SINOGRAM_SHAPE, Config.MASKS_DIR, args.num_masks)

if __name__ == "__main__":
    main()