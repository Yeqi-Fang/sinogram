"""
Training script for the SinogramTransformer model
"""
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

# Import from our modules
from models.sinogram_transformer import SinogramTransformer
from data.dataset import SinogramDataset, SinogramPatchDataset
from data.augmentation import SinogramAugmentation
from utils.metrics import calculate_psnr, calculate_ssim, calculate_rmse, AverageMeter
from utils.visualization import plot_comparison, plot_error_map, plot_training_curves
from config import Config

def setup_logging(log_dir):
    """Set up logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create file handler
    file_handler = logging.FileHandler(os.path.join(log_dir, 'train.log'))
    file_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def train_one_epoch(model, dataloader, criterion, optimizer, device, logger):
    """Train the model for one epoch"""
    model.train()
    losses = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    
    # Progress bar
    pbar = tqdm(dataloader, desc="Training")
    
    for inputs, targets in pbar:
        # Move data to device
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Calculate loss
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update statistics
        batch_size = inputs.size(0)
        losses.update(loss.item(), batch_size)
        
        # Calculate metrics for progress display
        with torch.no_grad():
            psnr_val = calculate_psnr(outputs.detach(), targets)
            ssim_val = calculate_ssim(outputs.detach(), targets)
            psnr_meter.update(psnr_val, batch_size)
            ssim_meter.update(ssim_val, batch_size)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{losses.avg:.4f}",
            'psnr': f"{psnr_meter.avg:.2f}",
            'ssim': f"{ssim_meter.avg:.4f}"
        })
    
    # Log epoch statistics
    logger.info(f"Train Loss: {losses.avg:.4f}, PSNR: {psnr_meter.avg:.2f}, SSIM: {ssim_meter.avg:.4f}")
    
    return losses.avg, psnr_meter.avg, ssim_meter.avg

def validate(model, dataloader, criterion, device, logger):
    """Validate the model"""
    model.eval()
    losses = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    rmse_meter = AverageMeter()
    
    # Progress bar
    pbar = tqdm(dataloader, desc="Validation")
    
    # Store a sample for visualization
    vis_sample = None
    
    with torch.no_grad():
        for inputs, targets in pbar:
            # Move data to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Update statistics
            batch_size = inputs.size(0)
            losses.update(loss.item(), batch_size)
            
            # Calculate metrics
            psnr_val = calculate_psnr(outputs, targets)
            ssim_val = calculate_ssim(outputs, targets)
            rmse_val = calculate_rmse(outputs, targets)
            
            psnr_meter.update(psnr_val, batch_size)
            ssim_meter.update(ssim_val, batch_size)
            rmse_meter.update(rmse_val, batch_size)
            
            # Store a sample for visualization
            if vis_sample is None:
                vis_sample = (inputs, outputs, targets)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses.avg:.4f}",
                'psnr': f"{psnr_meter.avg:.2f}",
                'ssim': f"{ssim_meter.avg:.4f}"
            })
    
    # Log validation statistics
    logger.info(f"Val Loss: {losses.avg:.4f}, PSNR: {psnr_meter.avg:.2f}, SSIM: {ssim_meter.avg:.4f}, RMSE: {rmse_meter.avg:.4f}")
    
    return losses.avg, psnr_meter.avg, ssim_meter.avg, rmse_meter.avg, vis_sample

def main():
    parser = argparse.ArgumentParser(description="Train the SinogramTransformer model")
    parser.add_argument("--data_path", type=str, default=Config.PROCESSED_SINOGRAM_PATH,
                       help="Path to the processed sinogram data")
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE,
                       help="Batch size for training")
    parser.add_argument("--lr", type=float, default=Config.LEARNING_RATE,
                       help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=Config.NUM_EPOCHS,
                       help="Number of epochs")
    parser.add_argument("--save_dir", type=str, default=Config.SAVE_DIR,
                       help="Directory to save checkpoints")
    parser.add_argument("--no_augmentation", action="store_true",
                       help="Disable data augmentation")
    parser.add_argument("--use_patches", action="store_true",
                       help="Use patch-based dataset")
    
    args = parser.parse_args()
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    log_dir = os.path.join(args.save_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    vis_dir = os.path.join(args.save_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Set up logging
    logger = setup_logging(log_dir)
    logger.info(f"Starting training with arguments: {args}")
    
    # Data augmentation is disabled
    augmentation = None
    logger.info("Data augmentation is disabled")
    
    # Create datasets
    if args.use_patches:
        logger.info("Using patch-based dataset")
        dataset = SinogramPatchDataset(
            args.data_path,
            mask_angles=(Config.MASK_ANGLE_START, Config.MASK_ANGLE_END),
            transform=augmentation,
            num_patches=5000  # Increase for more diverse training
        )
    else:
        logger.info("Using block-based dataset")
        dataset = SinogramDataset(
            args.data_path,
            mask_angles=(Config.MASK_ANGLE_START, Config.MASK_ANGLE_END),
            transform=augmentation,
            mask_paths=[os.path.join(Config.MASKS_DIR, f'mask_{i+1}.pt') for i in range(Config.NUM_MASKS)],
            use_predefined_masks=not args.no_augmentation and Config.USE_AUGMENTATION
        )
    
    # Split dataset into train and validation sets
    dataset_size = len(dataset)
    train_size = int(Config.TRAIN_VAL_SPLIT * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    logger.info(f"Dataset size: {dataset_size}, Train size: {train_size}, Val size: {val_size}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
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
    
    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.lr,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=5, factor=0.5, verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    val_metrics = {
        'psnr': [],
        'ssim': [],
        'rmse': []
    }
    
    start_time = time.time()
    
    for epoch in range(args.num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        # Train
        train_loss, train_psnr, train_ssim = train_one_epoch(
            model, train_loader, criterion, optimizer, device, logger
        )
        train_losses.append(train_loss)
        
        # Validate
        val_loss, val_psnr, val_ssim, val_rmse, vis_sample = validate(
            model, val_loader, criterion, device, logger
        )
        val_losses.append(val_loss)
        val_metrics['psnr'].append(val_psnr)
        val_metrics['ssim'].append(val_ssim)
        val_metrics['rmse'].append(val_rmse)
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Current learning rate: {current_lr:.2e}")
        
        # Save visualization
        if vis_sample is not None:
            inputs, outputs, targets = vis_sample
            fig = plot_comparison(
                inputs[0], outputs[0], targets[0],
                slice_idx=min(32, inputs.shape[3]-1) if inputs.dim() > 3 else 0
            )
            fig.savefig(os.path.join(vis_dir, f'comparison_epoch_{epoch+1}.png'))
            plt.close(fig)
            
            fig = plot_error_map(outputs[0], targets[0])
            fig.savefig(os.path.join(vis_dir, f'error_map_epoch_{epoch+1}.png'))
            plt.close(fig)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_psnr': val_psnr,
                'val_ssim': val_ssim,
                'val_rmse': val_rmse
            }, os.path.join(args.save_dir, 'best_model.pth'))
            logger.info(f"Saved best model with val_loss: {best_val_loss:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_psnr': val_psnr,
                'val_ssim': val_ssim,
                'val_rmse': val_rmse
            }, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
            logger.info(f"Saved checkpoint for epoch {epoch+1}")
    
    # Calculate total training time
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time/60:.2f} minutes")
    
    # Plot training curves
    fig = plot_training_curves(train_losses, val_losses, val_metrics)
    fig.savefig(os.path.join(args.save_dir, 'training_curves.png'))
    plt.close(fig)
    
    # Save final model
    torch.save({
        'epoch': args.num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
        'val_psnr': val_psnr,
        'val_ssim': val_ssim,
        'val_rmse': val_rmse
    }, os.path.join(args.save_dir, 'final_model.pth'))
    logger.info("Training completed!")

if __name__ == "__main__":
    main()