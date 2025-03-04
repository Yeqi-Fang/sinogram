"""
Unit tests for dataset classes
"""
import unittest
import torch
import os
import tempfile
from data.dataset import SinogramDataset, SinogramPatchDataset

class TestSinogramDataset(unittest.TestCase):
    """Test the SinogramDataset class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create temporary sinogram file
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.sinogram_path = os.path.join(self.tmp_dir.name, 'test_sinogram.pt')
        
        # Create a small test sinogram
        self.height = 32
        self.width = 64
        self.depth = 128
        self.sinogram = torch.rand((self.height, self.width, self.depth))
        torch.save(self.sinogram, self.sinogram_path)
        
        # Create a test mask
        self.mask_path = os.path.join(self.tmp_dir.name, 'test_mask.pt')
        self.mask = torch.ones((self.height, self.width, self.depth))
        self.mask[10:20, :, :] = 0  # Mask a section
        torch.save(self.mask, self.mask_path)
    
    def tearDown(self):
        """Tear down test fixtures"""
        self.tmp_dir.cleanup()
    
    def test_initialization(self):
        """Test that the dataset initializes correctly"""
        block_size = 32
        dataset = SinogramDataset(
            sinogram_path=self.sinogram_path,
            block_size=block_size
        )
        
        # Check number of blocks
        expected_blocks = self.depth // block_size
        if self.depth % block_size != 0:
            expected_blocks += 1
        
        self.assertEqual(len(dataset), expected_blocks,
                        f"Expected {expected_blocks} blocks, got {len(dataset)}")
    
    def test_getitem(self):
        """Test that __getitem__ returns correct data"""
        block_size = 32
        dataset = SinogramDataset(
            sinogram_path=self.sinogram_path,
            block_size=block_size
        )
        
        # Get first block
        block, target = dataset[0]
        
        # Check shapes
        expected_shape = (self.height, self.width, block_size)
        self.assertEqual(block.shape, expected_shape,
                        f"Expected shape {expected_shape}, got {block.shape}")
        self.assertEqual(target.shape, expected_shape,
                        f"Expected shape {expected_shape}, got {target.shape}")
        
        # Check that block and target are not identical (due to masking)
        self.assertTrue(torch.allclose(block, target),
                        "Block and target should be the same without masking")
    
    def test_masking(self):
        """Test that masking is applied correctly"""
        block_size = 32
        dataset = SinogramDataset(
            sinogram_path=self.sinogram_path,
            mask_angles=(30, 60),  # Mask 30-60 degrees
            block_size=block_size
        )
        
        # Get first block
        block, target = dataset[0]
        
        # Check that masking was applied
        self.assertFalse(torch.allclose(block, target),
                        "Block and target should differ due to masking")
        
        # Check that some rows are zeroed out
        zero_rows = torch.sum(block, dim=(1, 2)) == 0
        self.assertTrue(torch.any(zero_rows),
                       "Some rows should be zeroed out due to masking")
    
    def test_predefined_mask(self):
        """Test that predefined mask is applied correctly"""
        block_size = 32
        dataset = SinogramDataset(
            sinogram_path=self.sinogram_path,
            mask_paths=[self.mask_path],
            use_predefined_masks=True,
            block_size=block_size
        )
        
        # Get first block
        block, target = dataset[0]
        
        # Check that masking was applied
        self.assertFalse(torch.allclose(block, target),
                        "Block and target should differ due to masking")
        
        # Check that the expected rows are zeroed out
        zero_sum = torch.sum(block[10:20, :, :])
        self.assertEqual(zero_sum.item(), 0,
                        "Rows 10-20 should be zeroed out due to mask")

class TestSinogramPatchDataset(unittest.TestCase):
    """Test the SinogramPatchDataset class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create temporary sinogram file
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.sinogram_path = os.path.join(self.tmp_dir.name, 'test_sinogram.pt')
        
        # Create a small test sinogram
        self.height = 64
        self.width = 64
        self.depth = 128
        self.sinogram = torch.rand((self.height, self.width, self.depth))
        torch.save(self.sinogram, self.sinogram_path)
    
    def tearDown(self):
        """Tear down test fixtures"""
        self.tmp_dir.cleanup()
    
    def test_initialization(self):
        """Test that the dataset initializes correctly"""
        patch_size = (32, 32, 32)
        num_patches = 10
        dataset = SinogramPatchDataset(
            sinogram_path=self.sinogram_path,
            patch_size=patch_size,
            num_patches=num_patches
        )
        
        # Check number of patches
        self.assertEqual(len(dataset), num_patches,
                        f"Expected {num_patches} patches, got {len(dataset)}")
    
    def test_getitem(self):
        """Test that __getitem__ returns correct data"""
        patch_size = (32, 32, 32)
        dataset = SinogramPatchDataset(
            sinogram_path=self.sinogram_path,
            patch_size=patch_size,
            num_patches=10
        )
        
        # Get a patch
        patch, target = dataset[0]
        
        # Check shapes
        self.assertEqual(patch.shape, patch_size,
                        f"Expected shape {patch_size}, got {patch.shape}")
        self.assertEqual(target.shape, patch_size,
                        f"Expected shape {patch_size}, got {target.shape}")
        
        # Check that patch and target are the same without masking
        self.assertTrue(torch.allclose(patch, target),
                       "Patch and target should be the same without masking")
    
    def test_masking(self):
        """Test that masking is applied correctly"""
        patch_size = (32, 32, 32)
        dataset = SinogramPatchDataset(
            sinogram_path=self.sinogram_path,
            patch_size=patch_size,
            mask_angles=(30, 60),  # Mask 30-60 degrees
            num_patches=10
        )
        
        # Get a patch
        patch, target = dataset[0]
        
        # Check that masking was applied (patches shouldn't be identical)
        self.assertFalse(torch.allclose(patch, target),
                        "Patch and target should differ due to masking")
        
        # Check that some rows are zeroed out
        zero_rows = torch.sum(patch, dim=(1, 2)) == 0
        self.assertTrue(torch.any(zero_rows),
                       "Some rows should be zeroed out due to masking")

if __name__ == '__main__':
    unittest.main()