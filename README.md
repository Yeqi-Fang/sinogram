# Incomplete Ring PET Reconstruction

A deep learning framework for reconstructing complete PET sinograms from incomplete ring data. This project implements a transformer-based architecture that processes sinogram data in blocks to efficiently reconstruct missing angular information.

## Project Overview

Positron Emission Tomography (PET) imaging traditionally requires complete detector rings for high-quality reconstructions. However, due to hardware limitations, cost constraints, or specialized clinical needs, incomplete-ring PET scanners are increasingly common. This project addresses the challenge of reconstructing high-quality PET images from these incomplete detector setups.

The key innovation is the **SinogramTransformer** model, which:

1. Processes large 3D sinogram data efficiently in blocks
2. Uses a transformer architecture to capture relationships between different parts of the sinogram
3. Leverages spatial and contextual information to reconstruct missing angular data
4. Achieves high-quality reconstruction with relatively small computational overhead

## Repository Structure

```
project/
├── models/
│   ├── __init__.py
│   └── sinogram_transformer.py    # Main model architecture
├── data/
│   ├── __init__.py
│   ├── dataset.py                 # Dataset classes for sinograms
│   ├── utils.py                   # Data processing utilities
│   └── augmentation.py            # Augmentation techniques (disabled)
├── utils/
│   ├── __init__.py
│   ├── metrics.py                 # Evaluation metrics (PSNR, SSIM)
│   └── visualization.py           # Visualization functions
├── tests/
│   ├── test_model.py              # Unit tests for model components
│   ├── test_dataset.py            # Unit tests for dataset classes
│   └── test_utils.py              # Unit tests for utility functions
├── config.py                      # Configuration parameters
├── generate_specific_data.py      # Script to generate synthetic data
├── train_with_generated_data.py   # Training script
└── run_tests.py                   # Script to run all unit tests
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy
- matplotlib
- scikit-image
- tqdm

### Setup

```bash
# Clone the repository
git clone https://github.com/Yeqi-Fang/sinogram.git
cd sinogram

# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install torch numpy matplotlib scikit-image tqdm
```

## Usage

### Data Generation

Generate synthetic sinogram data for training and testing:

```bash
python generate_specific_data.py \
    --output_dir ./data/pet_dataset \
    --pattern brain \
    --num_processes 8
```

Parameters:
- `--output_dir`: Directory to save the generated data
- `--pattern`: Pattern type (brain, phantom)
- `--num_processes`: Number of parallel processes for data generation
- `--mask_angle_start`, `--mask_angle_end`: Angular range to mask (degrees)

### Training

Train the model using the generated data:

```bash
python train_with_generated_data_fixed.py \
    --data_dir ./data/pet_dataset \
    --output_dir ./outputs \
    --batch_size 1 \
    --num_epochs 30
```

Parameters:
- `--data_dir`: Directory containing training and testing data
- `--output_dir`: Directory to save model checkpoints and results
- `--batch_size`: Batch size for training (recommended 1 for large sinograms)
- `--num_epochs`: Number of training epochs
- `--lr`: Learning rate
- `--gradient_accumulation`: Steps for gradient accumulation

### Testing

The test functionality is integrated into the training script. After training, the model is evaluated on the test set. To test a trained model:

```bash
python test_model_complete.py \
    --data_path ./data/pet_dataset/complete_sinogram.pt \
    --save_dir ./test_results
```

## Model Architecture

The model consists of two main components:

1. **SinogramBlockEncoder**: Processes individual sinogram blocks using 2D convolutions
2. **SinogramTransformer**: Uses transformer layers to model relationships between blocks

The sinogram data (typically of size 112×225×1024) is processed in blocks along the third dimension. Each block is encoded, transformed through self-attention, and then decoded back to the original space.

Key features:
- Efficient memory usage through adaptive pooling
- Position embeddings to maintain spatial relationships
- Multi-head self-attention to capture global context
- Block-wise processing to handle large data volumes

## Technical Details

### Sinogram Data Structure

The project works with sinogram data in the following format:
- First dimension (height = 112): Angular samples
- Second dimension (width = 225): Radial samples
- Third dimension (depth = 1024): Ring differences or planes

### Incomplete Ring Simulation

We simulate incomplete ring PET by masking out specific angular sections of the sinogram (typically in the range of 30-60 degrees). This mimics the data loss that occurs when part of the detector ring is missing.

### Metrics

The model is evaluated using:
- Peak Signal-to-Noise Ratio (PSNR)
- Structural Similarity Index (SSIM)
- Mean Squared Error (MSE)

## Troubleshooting

### Memory Issues

If you encounter CUDA out of memory errors:
- Reduce batch size to 1
- Enable gradient accumulation with `--gradient_accumulation 4`
- Reduce model size with smaller `--embed_dim`

### SSIM Calculation Errors

The metrics functions handle 3D data by computing metrics on sampled 2D slices. If you encounter SSIM calculation errors, ensure you're using the patched metrics.py file which sets an appropriate window size for small images.


## License

This project is licensed under the MIT License - see the LICENSE file for details.