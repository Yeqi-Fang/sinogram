# Incomplete Ring PET Reconstruction System

This project implements a deep learning-based system for reconstructing PET images from incomplete sinogram data, where part of the detector ring is missing (typically in the range of 30-60 degrees). The method uses a Transformer-based model to process sinogram data in blocks, learning to fill in missing angular information while preserving structural details.

## Project Structure

```
project/
├── models/
│   ├── __init__.py
│   └── sinogram_transformer.py  # Transformer model for sinogram completion
├── data/
│   ├── __init__.py
│   ├── dataset.py               # Dataset classes for sinogram data
│   ├── utils.py                 # Data processing utilities
│   └── augmentation.py          # Data augmentation techniques
├── utils/
│   ├── __init__.py
│   ├── metrics.py               # Evaluation metrics (PSNR, SSIM, etc.)
│   └── visualization.py         # Visualization utilities
├── config.py                    # Configuration parameters
├── preprocess.py                # Data preprocessing script
├── train.py                     # Training script
└── test.py                      # Testing and evaluation script
```

## Model Architecture

The key innovation in this project is the `SinogramTransformer` model, which:

1. Processes sinogram data in blocks along the ring difference dimension
2. Uses a transformer-based architecture to capture contextual relationships between blocks
3. Combines CNNs for spatial feature extraction with transformer layers for contextual modeling
4. Leverages positional embeddings to maintain block order information

The model accepts incomplete sinogram data (with missing angular sections) as input and outputs the completed sinogram.

## Data Processing

Sinogram data in PET is structured as a 3D tensor:

- First dimension (H): Represents angles θ from 0-2π
- Second dimension (W): Represents spatial width (radial position)
- Third dimension (C): Represents ring differences (thousands of planes)

Since the third dimension is very large (typically 4096), processing the entire sinogram at once is computationally infeasible. Our approach:

1. Divides the sinogram into manageable blocks along the ring difference dimension
2. Processes each block with a shared encoder
3. Uses a transformer to capture contextual relationships between blocks
4. Reconstructs the output from the transformed blocks

## Usage

### Setup

1. Update the configuration in `config.py` with your dataset paths and desired parameters.
2. Ensure you have PyTorch and other dependencies installed.

### Preprocessing

```bash
python preprocess.py --sinogram_path /path/to/your/sinogram.pt --normalize
```

### Training

```bash
python train.py --data_path /path/to/processed_sinogram.pt --batch_size 4 --num_epochs 50
```

### Testing

```bash
python test.py --checkpoint ./checkpoints/best_model.pth --results_dir ./results
```

## Data Augmentation

The system includes multiple data augmentation techniques to improve model robustness:

- Random angular masking (simulating different missing sections)
- Additive noise (robustness to noise)
- Angular shifts (invariance to rotation)

## Evaluation Metrics

The model performance is evaluated using:

- Peak Signal-to-Noise Ratio (PSNR)
- Structural Similarity Index (SSIM)
- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)

## Visualization

The package includes comprehensive visualization tools:

- Sinogram slice visualization (before/after reconstruction)
- Error maps (spatial distribution of errors)
- Training curve plots (loss, metrics)
- Multi-slice visualization (for exploring different ring differences)

## References

1. PET Reconstruction Techniques
2. Transformer Models for Medical Imaging
3. PyTomography Documentation

## Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy
- Matplotlib
- scikit-image (for metrics)
- tqdm (for progress display)