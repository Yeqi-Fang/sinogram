"""
Generate synthetic sinogram data for testing PET reconstruction models.
This script creates realistic-looking sinogram data with configurable dimensions
and patterns to mimic PET scanner output, using only PyTorch operations to avoid
NumPy compatibility issues.
"""
import os
import torch
import math
import argparse
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
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

def generate_phantom_sinogram(height, width, depth, num_sources=5, noise_level=0.02):
    """
    Generate a synthetic sinogram with point sources.
    
    Args:
        height (int): Number of angular samples (theta)
        width (int): Number of radial samples (r)
        depth (int): Number of planes (ring differences)
        num_sources (int): Number of point sources to simulate
        noise_level (float): Standard deviation of Gaussian noise
        
    Returns:
        torch.Tensor: Synthetic sinogram
    """
    # Create empty sinogram
    sinogram = torch.zeros((height, width, depth))
    
    # Generate random point sources using torch's random functions
    sources = []
    for _ in range(num_sources):
        # Random position in image space
        x = torch.rand(1).item() * 1.5 - 0.75  # uniform [-0.75, 0.75]
        y = torch.rand(1).item() * 1.5 - 0.75  # uniform [-0.75, 0.75]
        # Random intensity (higher near center)
        intensity = (torch.rand(1).item() * 0.5 + 0.5) * (1.0 - 0.5 * (x**2 + y**2))
        # Random size
        size = torch.rand(1).item() * 0.15 + 0.05  # uniform [0.05, 0.2]
        sources.append((x, y, intensity, size))
    
    # For each angle, calculate projection
    angle_range = torch.linspace(0, math.pi, height)
    
    for angle_idx, angle in enumerate(angle_range):
        sin_theta = torch.sin(angle).item()
        cos_theta = torch.cos(angle).item()
        
        # For each radial position
        r_values = torch.linspace(-1, 1, width)
        
        for r_idx, r in enumerate(r_values):
            r = r.item()
            
            # For each source, calculate contribution
            for x, y, intensity, size in sources:
                # Project point onto line
                proj_dist = abs(x * cos_theta + y * sin_theta - r)
                
                # Calculate contribution (Gaussian profile)
                contrib = intensity * math.exp(-(proj_dist**2) / (2 * size**2))
                
                # Add to sinogram (for all depths with scaling)
                depth_scale = torch.linspace(1.0, 0.4, depth)
                for d in range(depth):
                    ring_diff = abs(d - depth // 2) / (depth // 2)  # Normalize ring difference
                    scale = math.exp(-ring_diff * 2)  # Exponential falloff with ring difference
                    sinogram[angle_idx, r_idx, d] += contrib * depth_scale[d].item() * scale
    
    # Add background activity
    background = torch.ones_like(sinogram) * 0.05
    sinogram = sinogram + background
    
    # Add noise
    noise = torch.randn_like(sinogram) * noise_level
    sinogram = sinogram + noise
    
    # Ensure non-negative values
    sinogram = torch.clamp(sinogram, min=0.0)
    
    # Normalize
    sinogram = normalize_sinogram(sinogram)
    
    return sinogram

def generate_ring_data(height, width, depth, num_rings=8, noise_level=0.02):
    """
    Generate synthetic sinogram with ring patterns.
    
    Args:
        height (int): Number of angular samples (theta)
        width (int): Number of radial samples (r)
        depth (int): Number of planes (ring differences)
        num_rings (int): Number of rings to generate
        noise_level (float): Standard deviation of Gaussian noise
        
    Returns:
        torch.Tensor: Synthetic sinogram
    """
    # Create empty sinogram
    sinogram = torch.zeros((height, width, depth))
    
    # Generate rings of different radii
    centers = []
    for _ in range(num_rings):
        # Random center position
        cx = torch.rand(1).item() - 0.5  # uniform [-0.5, 0.5]
        cy = torch.rand(1).item() - 0.5  # uniform [-0.5, 0.5]
        # Random radius
        radius = torch.rand(1).item() * 0.3 + 0.1  # uniform [0.1, 0.4]
        # Random intensity
        intensity = torch.rand(1).item() * 0.6 + 0.4  # uniform [0.4, 1.0]
        centers.append((cx, cy, radius, intensity))
    
    # Precompute angles
    angle_range = torch.linspace(0, math.pi, height)
    
    # For each angle, calculate projection
    for angle_idx, angle in enumerate(angle_range):
        sin_theta = torch.sin(angle).item()
        cos_theta = torch.cos(angle).item()
        
        # Create projection for each ring
        for cx, cy, radius, intensity in centers:
            # Calculate projected position of center
            center_proj = cx * cos_theta + cy * sin_theta
            
            # Calculate start and end points of ring projection
            start_proj = center_proj - radius
            end_proj = center_proj + radius
            
            # Convert to pixel indices
            start_idx = int((start_proj + 1) * (width - 1) / 2)
            end_idx = int((end_proj + 1) * (width - 1) / 2)
            
            # Clamp indices to valid range
            start_idx = max(0, min(width - 1, start_idx))
            end_idx = max(0, min(width - 1, end_idx))
            
            # Create intensity profile (higher at edges)
            if start_idx < end_idx:
                # Rectangular profile
                for r_idx in range(start_idx, end_idx + 1):
                    # Normalized position within ring
                    rel_pos = (r_idx - start_idx) / max(1, end_idx - start_idx)
                    # Intensity profile (higher at edges)
                    edge_factor = 4 * (rel_pos - 0.5)**2  # Parabolic profile
                    profile = intensity * (0.5 + 0.5 * edge_factor)
                    
                    # Add to sinogram with depth scaling
                    for d in range(depth):
                        ring_diff = abs(d - depth // 2) / (depth // 2)
                        scale = math.exp(-ring_diff * 2)
                        sinogram[angle_idx, r_idx, d] += profile * scale
    
    # Add background activity
    background = torch.ones_like(sinogram) * 0.08
    sinogram = sinogram + background
    
    # Add noise
    noise = torch.randn_like(sinogram) * noise_level
    sinogram = sinogram + noise
    
    # Ensure non-negative values
    sinogram = torch.clamp(sinogram, min=0.0)
    
    # Normalize
    sinogram = normalize_sinogram(sinogram)
    
    return sinogram

def generate_brain_like_sinogram(height, width, depth, noise_level=0.02):
    """
    Generate a synthetic sinogram resembling brain activity.
    
    Args:
        height (int): Number of angular samples (theta)
        width (int): Number of radial samples (r)
        depth (int): Number of planes (ring differences)
        noise_level (float): Standard deviation of Gaussian noise
        
    Returns:
        torch.Tensor: Synthetic sinogram
    """
    # Create empty sinogram
    sinogram = torch.zeros((height, width, depth))
    
    # Define brain-like structures
    structures = []
    
    # Outer skull (ring)
    structures.append(('ring', 0, 0, 0.8, 0.15, 0.4))  # type, cx, cy, radius, thickness, intensity
    
    # Brain tissue (filled circle)
    structures.append(('disk', 0, 0, 0.65, 0, 0.6))  # type, cx, cy, radius, _, intensity
    
    # Ventricles (darker regions)
    structures.append(('disk', 0.1, 0, 0.15, 0, -0.3))  # Negative intensity to subtract
    structures.append(('disk', -0.1, 0, 0.15, 0, -0.3))
    
    # Hot spots (small bright regions)
    for _ in range(10):
        # Random position within brain
        angle = torch.rand(1).item() * 2 * math.pi
        dist = torch.rand(1).item() * 0.5
        x = dist * math.cos(angle)
        y = dist * math.sin(angle)
        # Random size and intensity
        size = torch.rand(1).item() * 0.07 + 0.05  # uniform [0.05, 0.12]
        intensity = torch.rand(1).item() * 0.4 + 0.3  # uniform [0.3, 0.7]
        structures.append(('disk', x, y, size, 0, intensity))
    
    # Precompute angles
    angle_range = torch.linspace(0, math.pi, height)
    
    # For each angle, calculate projection
    for angle_idx, angle in enumerate(tqdm(angle_range, desc="Generating brain-like sinogram")):
        sin_theta = torch.sin(angle).item()
        cos_theta = torch.cos(angle).item()
        
        # For each radial position
        r_values = torch.linspace(-1, 1, width)
        
        for r_idx, r in enumerate(r_values):
            r = r.item()
            
            # Process each structure
            for structure in structures:
                struct_type, cx, cy, radius, thickness, intensity = structure
                
                # Project center onto line
                center_proj = cx * cos_theta + cy * sin_theta
                
                if struct_type == 'ring':
                    # Calculate start and end points of ring projection
                    inner_radius = radius - thickness/2
                    outer_radius = radius + thickness/2
                    
                    # Inner and outer edge projections
                    inner_start = center_proj - inner_radius
                    inner_end = center_proj + inner_radius
                    outer_start = center_proj - outer_radius
                    outer_end = center_proj + outer_radius
                    
                    # Check if current r is within the projected ring
                    if (outer_start <= r <= inner_start) or (inner_end <= r <= outer_end):
                        # Distance from r to nearest inner edge
                        if r <= center_proj:
                            edge_dist = min(abs(r - outer_start), abs(r - inner_start))
                        else:
                            edge_dist = min(abs(r - inner_end), abs(r - outer_end))
                        
                        # Intensity profile
                        profile = intensity * (1 - edge_dist / thickness)
                        
                        # Add to all depths with scaling
                        for d in range(depth):
                            ring_diff = abs(d - depth // 2) / (depth // 2)
                            scale = math.exp(-ring_diff * 2)
                            sinogram[angle_idx, r_idx, d] += max(0, profile * scale)
                
                elif struct_type == 'disk':
                    # Distance from projection line to center
                    dist_to_center = abs(center_proj - r)
                    
                    # Check if current r is within the projected disk
                    if dist_to_center <= radius:
                        # Calculate profile based on distance from center
                        edge_factor = 1 - (dist_to_center / radius)**2
                        profile = intensity * edge_factor
                        
                        # Add to all depths with scaling
                        for d in range(depth):
                            ring_diff = abs(d - depth // 2) / (depth // 2)
                            scale = math.exp(-ring_diff * 2)
                            sinogram[angle_idx, r_idx, d] += profile * scale
    
    # Add background activity
    background = torch.ones_like(sinogram) * 0.1
    sinogram = sinogram + background
    
    # Add noise
    noise = torch.randn_like(sinogram) * noise_level
    sinogram = sinogram + noise
    
    # Ensure non-negative values
    sinogram = torch.clamp(sinogram, min=0.0)
    
    # Normalize
    sinogram = normalize_sinogram(sinogram)
    
    return sinogram

def create_incomplete_sinograms(sinogram, mask_angles=None, num_variants=1):
    """
    Create incomplete sinograms by masking angular regions.
    
    Args:
        sinogram (torch.Tensor): Complete sinogram
        mask_angles (list): List of (start_angle, end_angle) tuples in degrees
        num_variants (int): Number of variants to generate
        
    Returns:
        list: List of incomplete sinograms
    """
    if mask_angles is None:
        # Default mask angles
        mask_angles = [(30, 60), (45, 75), (135, 150), (20, 40)]
    
    incomplete_sinograms = []
    
    for i in range(num_variants):
        # Select a mask angle pair
        start_angle, end_angle = mask_angles[i % len(mask_angles)]
        
        # Create mask
        mask = create_angle_mask(sinogram.shape, start_angle, end_angle)
        
        # Apply mask
        masked_sinogram = apply_mask(sinogram, mask)
        
        incomplete_sinograms.append((masked_sinogram, (start_angle, end_angle)))
    
    return incomplete_sinograms

def visualize_sinograms(complete_sinogram, incomplete_sinograms, output_dir, slice_indices=None):
    """
    Visualize complete and incomplete sinograms.
    
    Args:
        complete_sinogram (torch.Tensor): Complete sinogram
        incomplete_sinograms (list): List of (incomplete_sinogram, mask_angles) tuples
        output_dir (str): Output directory
        slice_indices (list): List of slice indices to visualize
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if slice_indices is None:
        # Select some slice indices
        depth = complete_sinogram.shape[2]
        slice_indices = [0, depth//4, depth//2, 3*depth//4, depth-1]
        slice_indices = [min(i, depth-1) for i in slice_indices]
    
    # Visualize complete sinogram
    for i, slice_idx in enumerate(slice_indices):
        plt.figure(figsize=(10, 8))
        plt.subplot(len(incomplete_sinograms) + 1, 1, 1)
        plt.imshow(complete_sinogram[:, :, slice_idx].cpu().numpy(), cmap='hot', aspect='auto')
        plt.title(f"Complete Sinogram (Slice {slice_idx})")
        plt.colorbar()
        
        # Visualize incomplete sinograms
        for j, (masked_sino, mask_angles) in enumerate(incomplete_sinograms):
            plt.subplot(len(incomplete_sinograms) + 1, 1, j + 2)
            plt.imshow(masked_sino[:, :, slice_idx].cpu().numpy(), cmap='hot', aspect='auto')
            plt.title(f"Masked Sinogram ({mask_angles[0]}-{mask_angles[1]}°)")
            plt.colorbar()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"sinogram_comparison_slice_{slice_idx}.png"))
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic sinogram data for PET reconstruction")
    parser.add_argument("--output_dir", type=str, default="./data/synthetic",
                       help="Output directory for generated data")
    parser.add_argument("--height", type=int, default=180,
                       help="Number of angular samples")
    parser.add_argument("--width", type=int, default=256,
                       help="Number of radial samples")
    parser.add_argument("--depth", type=int, default=128,
                       help="Number of planes")
    parser.add_argument("--block_size", type=int, default=32,
                       help="Block size for data division")
    parser.add_argument("--noise_level", type=float, default=0.02,
                       help="Noise level for synthetic data")
    parser.add_argument("--pattern", type=str, default="brain",
                       choices=["phantom", "rings", "brain"],
                       help="Pattern to generate")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Generating synthetic sinogram data with shape: ({args.height}, {args.width}, {args.depth})")
    
    # Generate complete sinogram based on selected pattern
    if args.pattern == "phantom":
        complete_sinogram = generate_phantom_sinogram(
            args.height, args.width, args.depth, noise_level=args.noise_level
        )
    elif args.pattern == "rings":
        complete_sinogram = generate_ring_data(
            args.height, args.width, args.depth, noise_level=args.noise_level
        )
    else:  # brain
        complete_sinogram = generate_brain_like_sinogram(
            args.height, args.width, args.depth, noise_level=args.noise_level
        )
    
    print(f"Generated complete sinogram with shape: {complete_sinogram.shape}")
    
    # Save complete sinogram
    torch.save(complete_sinogram, os.path.join(args.output_dir, "complete_sinogram.pt"))
    print(f"Saved complete sinogram to {os.path.join(args.output_dir, 'complete_sinogram.pt')}")
    
    # Create visualization directory
    vis_dir = os.path.join(args.output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Visualize slices of the complete sinogram
    depth = complete_sinogram.shape[2]
    slice_indices = [0, depth//4, depth//2, 3*depth//4, depth-1]
    slice_indices = [min(i, depth-1) for i in slice_indices]
    
    for slice_idx in slice_indices:
        plt.figure(figsize=(10, 8))
        plt.imshow(complete_sinogram[:, :, slice_idx].numpy(), cmap='hot', aspect='auto')
        plt.title(f"Complete Sinogram (Slice {slice_idx})")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f"sinogram_slice_{slice_idx}.png"))
        plt.close()
    
    print(f"Saved visualizations to {vis_dir}")
    
    # Create training and testing datasets using block division
    train_dir = os.path.join(args.output_dir, "train")
    test_dir = os.path.join(args.output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Randomly split the sinogram for training and testing
    indices = np.random.permutation(depth)
    train_indices = indices[:int(0.8 * depth)]
    test_indices = indices[int(0.8 * depth):]
    
    # Create training dataset
    train_complete = complete_sinogram[:, :, train_indices]
    torch.save(train_complete, os.path.join(train_dir, "complete_sinogram.pt"))
    
    # Create testing dataset
    test_complete = complete_sinogram[:, :, test_indices]
    torch.save(test_complete, os.path.join(test_dir, "complete_sinogram.pt"))
    
    # Create blocks for training and testing
    block_size = min(args.block_size, depth)
    
    # Create and save blocks for training
    train_blocks = []
    for i in range(0, train_complete.shape[2], block_size):
        end_idx = min(i + block_size, train_complete.shape[2])
        block = train_complete[:, :, i:end_idx]
        
        # Pad if needed
        if block.shape[2] < block_size:
            padded_block = torch.zeros(
                (train_complete.shape[0], train_complete.shape[1], block_size),
                dtype=train_complete.dtype
            )
            padded_block[:, :, :block.shape[2]] = block
            block = padded_block
        
        train_blocks.append(block)
        torch.save(block, os.path.join(train_dir, f"block_{i//block_size+1}.pt"))
    
    # Create and save blocks for testing
    test_blocks = []
    for i in range(0, test_complete.shape[2], block_size):
        end_idx = min(i + block_size, test_complete.shape[2])
        block = test_complete[:, :, i:end_idx]
        
        # Pad if needed
        if block.shape[2] < block_size:
            padded_block = torch.zeros(
                (test_complete.shape[0], test_complete.shape[1], block_size),
                dtype=test_complete.dtype
            )
            padded_block[:, :, :block.shape[2]] = block
            block = padded_block
        
        test_blocks.append(block)
        torch.save(block, os.path.join(test_dir, f"block_{i//block_size+1}.pt"))
    
    print(f"Created {len(train_blocks)} training blocks and {len(test_blocks)} testing blocks")
    print(f"Saved training blocks to {train_dir}")
    print(f"Saved testing blocks to {test_dir}")
    
    print("Done!")

if __name__ == "__main__":
    main()