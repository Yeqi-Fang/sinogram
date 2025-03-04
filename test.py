"""
Testing script for the SinogramTransformer model
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import time

# Import from our modules
from models.sinogram_transformer import SinogramTransformer
from data.dataset import SinogramDataset, SinogramPatchDataset
from data.utils import apply_mask, create_angle_mask, combine_sinogram_blocks
from utils.metrics import calculate_metrics, AverageMeter
from utils.visualization import (
    plot_comparison, plot_error_map, visualize_multiple_slices
)
from config import Config

def setup_logging(log_dir):
    """Set up logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('test')
    logger.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create file handler
    file_handler = logging.FileHandler(os.path.join(log_dir, 'test.log'))
    file_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def test_model(model, dataloader, device, logger, results_dir):
    """Test the model and save results"""
    model.eval()
    
    # Metrics
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    rmse_meter = AverageMeter()
    mae_meter = AverageMeter()
    
    # Create directories for saving results
    os.makedirs(os.path.join(results_dir, 'comparisons'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'error_maps'), exist_ok=True)
    
    # Progress bar
    pbar = tqdm(dataloader, desc="Testing")
    
    # Save predictions for later reconstruction
    all_inputs = []
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(pbar):
            # Move data to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate metrics
            metrics = calculate_metrics(outputs, targets)
            
            # Update metrics
            batch_size = inputs.size(0)
            psnr_meter.update(metrics['psnr'], batch_size)
            ssim_meter.update(metrics['ssim'], batch_size)
            rmse_meter.update(metrics['rmse'], batch_size)
            mae_meter.update(metrics['mae'], batch_size)
            
            # Save predictions for later reconstruction
            all_inputs.append(inputs.cpu())
            all_outputs.append(outputs.cpu())
            all_targets.append(targets.cpu())
            
            # Update progress bar
            pbar.set_postfix({
                'psnr': f"{psnr_meter.avg:.2f}",
                'ssim': f"{ssim_meter.avg:.4f}"
            })
            
            # Save visualizations for the first few batches
            if i < 5:
                for j in range(min(batch_size, 2)):  # Save only first 2 samples per batch
                    # Plot comparison
                    fig = plot_comparison(
                        inputs[j], outputs[j], targets[j],
                        slice_idx=min(32, inputs.shape[3]-1) if inputs.dim() > 3 else 0
                    )
                    fig.savefig(os.path.join(
                        results_dir, 'comparisons', f'comparison_batch_{i+1}_sample_{j+1}.png'
                    ))
                    plt.close(fig)
                    
                    # Plot error map
                    fig = plot_error_map(outputs[j], targets[j])
                    fig.savefig(os.path.join(
                        results_dir, 'error_maps', f'error_map_batch_{i+1}_sample_{j+1}.png'
                    ))
                    plt.close(fig)
    
    # Log overall metrics
    logger.info(f"Test Results:")
    logger.info(f"  PSNR: {psnr_meter.avg:.4f}")
    logger.info(f"  SSIM: {ssim_meter.avg:.4f}")
    logger.info(f"  RMSE: {rmse_meter.avg:.6f}")
    logger.info(f"  MAE: {rmse_meter.avg:.6f}")
    
    # Save metrics to file
    with open(os.path.join(results_dir, 'metrics.txt'), 'w') as f:
        f.write(f"PSNR: {psnr_meter.avg:.4f}\n")
        f.write(f"SSIM: {ssim_meter.avg:.4f}\n")
        f.write(f"RMSE: {rmse_meter.avg:.6f}\n")
        f.write(f"MAE: {mae_meter.avg:.6f}\n")
    
    # Return metrics and predictions
    metrics = {
        'psnr': psnr_meter.avg,
        'ssim': ssim_meter.avg,
        'rmse': rmse_meter.avg,
        'mae': mae_meter.avg
    }
    
    return metrics, (all_inputs, all_outputs, all_targets)

def reconstruct_full_sinogram(predictions, original_shape, results_dir, logger):
    """
    Reconstruct full sinogram from block predictions
    
    Args:
        predictions (tuple): Tuple of (inputs, outputs, targets)
        original_shape (tuple): Original shape of the full sinogram
        results_dir (str): Directory to save results
        logger (logging.Logger): Logger instance
    """
    all_inputs, all_outputs, all_targets = predictions
    
    # Concatenate all batch predictions
    inputs = torch.cat([batch for batch in all_inputs], dim=0)
    outputs = torch.cat([batch for batch in all_outputs], dim=0)
    targets = torch.cat([batch for batch in all_targets], dim=0)
    
    logger.info(f"Reconstructing full sinogram from {inputs.shape[0]} blocks")
    
    # Convert to list of blocks if we're using patch-based dataset
    if inputs.dim() == 4:  # [B, H, W, C]
        inputs_blocks = [inputs[i] for i in range(inputs.shape[0])]
        outputs_blocks = [outputs[i] for i in range(outputs.shape[0])]
        targets_blocks = [targets[i] for i in range(targets.shape[0])]
    else:
        # Data is already in blocks
        inputs_blocks = inputs
        outputs_blocks = outputs
        targets_blocks = targets
    
    # Combine blocks to form full sinogram
    input_sinogram = combine_sinogram_blocks(inputs_blocks, original_shape[2])
    output_sinogram = combine_sinogram_blocks(outputs_blocks, original_shape[2])
    target_sinogram = combine_sinogram_blocks(targets_blocks, original_shape[2])
    
    logger.info(f"Reconstructed sinogram shapes:")
    logger.info(f"  Input: {input_sinogram.shape}")
    logger.info(f"  Output: {output_sinogram.shape}")
    logger.info(f"  Target: {target_sinogram.shape}")
    
    # Save reconstructed sinograms
    torch.save(input_sinogram, os.path.join(results_dir, 'input_sinogram.pt'))
    torch.save(output_sinogram, os.path.join(results_dir, 'output_sinogram.pt'))
    torch.save(target_sinogram, os.path.join(results_dir, 'target_sinogram.pt'))
    
    # Visualize multiple slices
    logger.info("Generating visualizations of reconstructed sinograms")
    
    # Visualize input sinogram
    fig = visualize_multiple_slices(
        input_sinogram, num_slices=8, start_idx=0, 
        step=input_sinogram.shape[2]//8, 
        title='Input Incomplete Sinogram'
    )
    fig.savefig(os.path.join(results_dir, 'input_sinogram_slices.png'))
    plt.close(fig)
    
    # Visualize output sinogram
    fig = visualize_multiple_slices(
        output_sinogram, num_slices=8, start_idx=0, 
        step=output_sinogram.shape[2]//8, 
        title='Reconstructed Sinogram'
    )
    fig.savefig(os.path.join(results_dir, 'output_sinogram_slices.png'))
    plt.close(fig)
    
    # Visualize target sinogram
    fig = visualize_multiple_slices(
        target_sinogram, num_slices=8, start_idx=0, 
        step=target_sinogram.shape[2]//8, 
        title='Ground Truth Sinogram'
    )
    fig.savefig(os.path.join(results_dir, 'target_sinogram_slices.png'))
    plt.close(fig)
    
    # Calculate metrics on full sinogram
    metrics = calculate_metrics(output_sinogram, target_sinogram)
    logger.info(f"Full Sinogram Metrics:")
    logger.info(f"  PSNR: {metrics['psnr']:.4f}")
    logger.info(f"  SSIM: {metrics['ssim']:.4f}")
    logger.info(f"  RMSE: {metrics['rmse']:.6f}")
    logger.info(f"  MAE: {metrics['mae']:.6f}")
    
    # Save metrics to file
    with open(os.path.join(results_dir, 'full_sinogram_metrics.txt'), 'w') as f:
        f.write(f"PSNR: {metrics['psnr']:.4f}\n")
        f.write(f"SSIM: {metrics['ssim']:.4f}\n")
        f.write(f"RMSE: {metrics['rmse']:.6f}\n")
        f.write(f"MAE: {metrics['mae']:.6f}\n")
    
    return input_sinogram, output_sinogram, target_sinogram

def main():
    parser = argparse.ArgumentParser(description="Test the SinogramTransformer model")
    parser.add_argument("--checkpoint", type=str, default=os.path.join(Config.SAVE_DIR, Config.TEST_CHECKPOINT),
                       help="Path to the model checkpoint")
    parser.add_argument("--data_path", type=str, default=Config.PROCESSED_SINOGRAM_PATH,
                       help="Path to the processed sinogram data")
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE,
                       help="Batch size for testing")
    parser.add_argument("--results_dir", type=str, default=Config.RESULTS_DIR,
                       help="Directory to save results")
    parser.add_argument("--use_patches", action="store_true",
                       help="Use patch-based dataset")
    parser.add_argument("--test_single_block", action="store_true",
                       help="Test only a single block (for debugging)")
    parser.add_argument("--mask_angle_start", type=float, default=Config.MASK_ANGLE_START,
                       help="Start angle for masking (degrees)")
    parser.add_argument("--mask_angle_end", type=float, default=Config.MASK_ANGLE_END,
                       help="End angle for masking (degrees)")
    
    args = parser.parse_args()
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create result directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Set up logging
    logger = setup_logging(args.results_dir)
    logger.info(f"Starting testing with arguments: {args}")
    
    # Create dataset
    if args.use_patches:
        logger.info("Using patch-based dataset")
        dataset = SinogramPatchDataset(
            args.data_path,
            mask_angles=(args.mask_angle_start, args.mask_angle_end),
            transform=None,
            num_patches=100  # Smaller for testing
        )
    else:
        logger.info("Using block-based dataset")
        dataset = SinogramDataset(
            args.data_path,
            mask_angles=(args.mask_angle_start, args.mask_angle_end),
            transform=None,
            block_size=Config.BLOCK_CHANNELS
        )
    
    # If testing only a single block
    if args.test_single_block:
        dataset = torch.utils.data.Subset(dataset, [0])
        logger.info("Testing only a single block")
    
    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    # Initialize model
    model = SinogramTransformer(
        sinogram_shape=Config.SINOGRAM_SHAPE,
        block_channels=Config.BLOCK_CHANNELS,
        embed_dim=Config.EMBED_DIM,
        num_heads=Config.NUM_HEADS,
        pool_size=Config.POOL_SIZE
    ).to(device)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Log additional checkpoint info if available
    if 'val_loss' in checkpoint:
        logger.info(f"Checkpoint validation loss: {checkpoint['val_loss']:.4f}")
    if 'val_psnr' in checkpoint:
        logger.info(f"Checkpoint validation PSNR: {checkpoint['val_psnr']:.2f}")
    if 'val_ssim' in checkpoint:
        logger.info(f"Checkpoint validation SSIM: {checkpoint['val_ssim']:.4f}")
    
    # Test model
    start_time = time.time()
    metrics, predictions = test_model(model, dataloader, device, logger, args.results_dir)
    test_time = time.time() - start_time
    logger.info(f"Testing completed in {test_time:.2f} seconds")
    
    # Reconstruct full sinogram
    logger.info("Reconstructing full sinogram")
    input_sinogram, output_sinogram, target_sinogram = reconstruct_full_sinogram(
        predictions, Config.SINOGRAM_SHAPE, args.results_dir, logger
    )
    
    logger.info("Testing completed!")

if __name__ == "__main__":
    main()