"""
Configuration parameters for the PET reconstruction model
"""
import os

class Config:
    # Data parameters
    SINOGRAM_PATH = '/path/to/sinogram.pt'  # Update with your actual path
    PROCESSED_SINOGRAM_PATH = './data/processed/sinogram_processed.pt'
    MASKS_DIR = './data/masks'
    NUM_MASKS = 5
    MASK_ANGLE_START = 30  # degrees
    MASK_ANGLE_END = 60    # degrees
    BLOCK_CHANNELS = 32    # Number of channels per block
    
    # Dataset parameters
    TRAIN_VAL_SPLIT = 0.8  # 80% train, 20% validation
    NUM_WORKERS = 4
    
    # Model parameters
    SINOGRAM_SHAPE = (224, 449, 4096)  # Shape of the complete sinogram (H, W, C)
    EMBED_DIM = 128
    NUM_HEADS = 8
    POOL_SIZE = (4, 4)
    
    # Augmentation parameters
    USE_AUGMENTATION = False  # Disabled data augmentation
    ROTATE_PROB = 0.0
    NOISE_PROB = 0.0
    MASK_PROB = 0.0
    NOISE_LEVEL = 0.0
    
    # Training parameters
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    NUM_EPOCHS = 50
    SAVE_DIR = './checkpoints'
    
    # Testing parameters
    TEST_CHECKPOINT = 'best_model.pth'
    RESULTS_DIR = './results'