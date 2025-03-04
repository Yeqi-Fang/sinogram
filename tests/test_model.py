"""
Unit tests for model architecture
"""
import unittest
import torch
from models.sinogram_transformer import SinogramTransformer, SinogramBlockEncoder, SinogramBlockDecoder

class TestSinogramBlockEncoder(unittest.TestCase):
    """Test the encoder component"""
    
    def test_forward(self):
        """Test that the forward pass produces correct output shape"""
        batch_size = 4
        height = 112
        width = 224
        in_channels = 32
        out_channels = 64
        pool_size = (4, 4)
        
        # Create encoder
        encoder = SinogramBlockEncoder(
            in_channels=in_channels,
            out_channels=out_channels,
            pool_size=pool_size
        )
        
        # Create input tensor [B, H, W, C]
        x = torch.randn(batch_size, height, width, in_channels)
        
        # Forward pass
        encoded = encoder(x)
        
        # Check output shape
        expected_shape = (batch_size, out_channels, pool_size[0] * pool_size[1])
        self.assertEqual(encoded.shape, expected_shape, 
                         f"Expected shape {expected_shape}, got {encoded.shape}")

class TestSinogramBlockDecoder(unittest.TestCase):
    """Test the decoder component"""
    
    def test_forward(self):
        """Test that the forward pass produces correct output shape"""
        batch_size = 4
        height = 112
        width = 224
        in_channels = 64
        out_channels = 32
        pool_size = (4, 4)
        
        # Create decoder
        decoder = SinogramBlockDecoder(
            in_channels=in_channels,
            out_channels=out_channels,
            pool_size=pool_size
        )
        
        # Create input tensor [B, C, pool_size[0]*pool_size[1]]
        x = torch.randn(batch_size, in_channels, pool_size[0] * pool_size[1])
        
        # Forward pass
        decoded = decoder(x, (height, width))
        
        # Check output shape
        expected_shape = (batch_size, height, width, out_channels)
        self.assertEqual(decoded.shape, expected_shape,
                        f"Expected shape {expected_shape}, got {decoded.shape}")

class TestSinogramTransformer(unittest.TestCase):
    """Test the full transformer model"""
    
    def test_initialization(self):
        """Test that the model initializes correctly"""
        sinogram_shape = (112, 224, 1024)
        block_channels = 32
        
        # Initialize model
        model = SinogramTransformer(
            sinogram_shape=sinogram_shape,
            block_channels=block_channels
        )
        
        # Check number of blocks
        expected_blocks = sinogram_shape[2] // block_channels
        self.assertEqual(model.num_blocks, expected_blocks,
                        f"Expected {expected_blocks} blocks, got {model.num_blocks}")
    
    def test_forward(self):
        """Test that the forward pass produces correct output shape"""
        batch_size = 2
        height = 112
        width = 224
        total_channels = 1024
        block_channels = 32
        
        # Initialize model
        model = SinogramTransformer(
            sinogram_shape=(height, width, total_channels),
            block_channels=block_channels,
            embed_dim=64,
            num_heads=4,
            pool_size=(2, 2)
        )
        
        # Create input tensor [B, H, W, C]
        x = torch.randn(batch_size, height, width, total_channels)
        
        # Forward pass
        output = model(x)
        
        # Check output shape
        expected_shape = (batch_size, height, width, total_channels)
        self.assertEqual(output.shape, expected_shape,
                        f"Expected shape {expected_shape}, got {output.shape}")
    
    def test_block_processing(self):
        """Test that the model correctly processes blocks"""
        batch_size = 2
        height = 64
        width = 64
        total_channels = 128
        block_channels = 32
        
        # Initialize model
        model = SinogramTransformer(
            sinogram_shape=(height, width, total_channels),
            block_channels=block_channels,
            embed_dim=64,
            num_heads=4,
            pool_size=(2, 2)
        )
        
        # Create input with recognizable pattern
        x = torch.zeros(batch_size, height, width, total_channels)
        # Make each block unique
        for i in range(total_channels // block_channels):
            x[:, :, :, i*block_channels:(i+1)*block_channels] = i + 1
        
        # Forward pass
        output = model(x)
        
        # Check that output has same dimensions
        self.assertEqual(output.shape, x.shape,
                        f"Expected shape {x.shape}, got {output.shape}")
        
        # Check that output is not identical to input (model should transform)
        self.assertFalse(torch.allclose(output, x, atol=1e-3),
                         "Output should not be identical to input")

if __name__ == '__main__':
    unittest.main()