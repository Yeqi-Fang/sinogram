"""
Unit tests for data augmentation
"""
import unittest
import torch
from data.augmentation import (
    SinogramAugmentation, SinogramMixup, RandomAngularMasking
)

class TestSinogramAugmentation(unittest.TestCase):
    """Test SinogramAugmentation class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sinogram = torch.ones((60, 40, 20))  # (angles, radial, planes)
    
    def test_rotation(self):
        """Test rotation augmentation"""
        # Force rotation to always occur
        augmentation = SinogramAugmentation(rotate_prob=1.0, noise_prob=0, mask_prob=0)
        
        # Apply augmentation
        augmented = augmentation(self.sinogram)
        
        # Check that shape is preserved
        self.assertEqual(augmented.shape, self.sinogram.shape,
                        "Augmentation should preserve shape")
        
        # Check that augmentation was applied (data changed)
        self.assertFalse(torch.allclose(augmented, self.sinogram),
                        "Augmentation should modify data")
    
    def test_noise(self):
        """Test noise augmentation"""
        # Force noise to always occur
        augmentation = SinogramAugmentation(rotate_prob=0, noise_prob=1.0, mask_prob=0, noise_level=0.1)
        
        # Apply augmentation
        augmented = augmentation(self.sinogram)
        
        # Check that shape is preserved
        self.assertEqual(augmented.shape, self.sinogram.shape,
                        "Augmentation should preserve shape")
        
        # Check that augmentation was applied (data changed)
        self.assertFalse(torch.allclose(augmented, self.sinogram),
                        "Augmentation should modify data")
        
        # Check that noise was added (values differ by expected amount)
        diff = (augmented - self.sinogram).abs()
        self.assertLess(diff.mean().item(), 0.2,
                       "Noise level should be reasonable")
    
    def test_masking(self):
        """Test angular masking augmentation"""
        # Force masking to always occur
        augmentation = SinogramAugmentation(rotate_prob=0, noise_prob=0, mask_prob=1.0)
        
        # Apply augmentation
        augmented = augmentation(self.sinogram)
        
        # Check that shape is preserved
        self.assertEqual(augmented.shape, self.sinogram.shape,
                        "Augmentation should preserve shape")
        
        # Check that some rows are zeroed out
        zero_rows = torch.sum(augmented, dim=(1, 2)) == 0
        self.assertTrue(torch.any(zero_rows),
                       "Some rows should be zeroed out due to masking")

class TestSinogramMixup(unittest.TestCase):
    """Test SinogramMixup class"""
    
    def test_mixup(self):
        """Test mixup augmentation"""
        batch_size = 4
        height = 30
        width = 40
        channels = 20
        
        # Create test batch
        inputs = torch.ones((batch_size, height, width, channels))
        targets = torch.zeros((batch_size, height, width, channels))
        batch = (inputs, targets)
        
        # Apply mixup
        mixup = SinogramMixup(alpha=0.5)
        mixed_inputs, mixed_targets = mixup(batch)
        
        # Check shapes
        self.assertEqual(mixed_inputs.shape, inputs.shape,
                        "Mixup should preserve input shape")
        self.assertEqual(mixed_targets.shape, targets.shape,
                        "Mixup should preserve target shape")
        
        # Check that mixup was applied
        # Since we mixed ones and zeros, values should be between 0 and 1
        self.assertTrue(torch.all(mixed_inputs >= 0) and torch.all(mixed_inputs <= 1),
                       "Mixed inputs should have values between 0 and 1")
        self.assertTrue(torch.all(mixed_targets >= 0) and torch.all(mixed_targets <= 1),
                       "Mixed targets should have values between 0 and 1")

class TestRandomAngularMasking(unittest.TestCase):
    """Test RandomAngularMasking class"""
    
    def test_masking(self):
        """Test random angular masking"""
        height = 60
        width = 40
        depth = 20
        sinogram = torch.ones((height, width, depth))
        
        # Apply masking
        masking = RandomAngularMasking(min_angle=10, max_angle=30, num_sections=1)
        masked = masking(sinogram)
        
        # Check shape
        self.assertEqual(masked.shape, sinogram.shape,
                        "Masking should preserve shape")
        
        # Check that some rows are zeroed out
        zero_rows = torch.sum(masked, dim=(1, 2)) == 0
        self.assertTrue(torch.any(zero_rows),
                       "Some rows should be zeroed out due to masking")
        
        # Check that not too many rows are zeroed out
        # Max angle is 30 degrees, which corresponds to 10 rows in a 60-row sinogram
        self.assertLessEqual(torch.sum(zero_rows).item(), height / 3,
                            "Number of masked rows should not exceed 1/3 of total rows")

if __name__ == '__main__':
    unittest.main()