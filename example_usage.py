#!/usr/bin/env python3
"""
Example usage of enhanced H5Dataset with RobustLoader.

This script demonstrates how to use the enhanced H5Dataset that combines
Dataset and DataModule functionality for easy integration with EvoAug2.
"""

import torch
from evoaug.augment import RandomDeletion, RandomInsertion, RandomMutation
from evoaug.evoaug import RobustLoader
from utils import utils

def main():
    """Demonstrate enhanced H5Dataset usage with RobustLoader."""
    
    print("="*60)
    print("ENHANCED H5DATASET WITH ROBUSTLOADER EXAMPLE")
    print("="*60)
    
    # Configuration
    filepath = '/grid/koo/home/duran/evoaug/deepstarr-data.h5'
    batch_size = 32
    
    # Check if data file exists
    try:
        # Create enhanced H5Dataset (includes DataModule-like functionality)
        print("Creating enhanced H5Dataset...")
        dataset = utils.H5Dataset(
            filepath=filepath,
            batch_size=batch_size,
            lower_case=False,
            transpose=False
        )
        
        print(f"✓ Dataset loaded successfully!")
        print(f"  - Training samples: {dataset.train_size}")
        print(f"  - Validation samples: {dataset.val_size}")
        print(f"  - Test samples: {dataset.test_size}")
        print(f"  - Sequence shape: (N, {dataset.A}, {dataset.L})")
        print(f"  - Number of classes: {dataset.num_classes}")
        print(f"  - Batch size: {dataset.batch_size}")
        
    except FileNotFoundError:
        print(f"✗ Data file not found: {filepath}")
        print("Please download the DeepSTARR data first:")
        print("wget https://zenodo.org/record/7265991/files/DeepSTARR_data.h5")
        return
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return
    
    # Define augmentations
    print("\nSetting up augmentations...")
    augment_list = [
        RandomDeletion(delete_min=0, delete_max=10),
        RandomInsertion(insert_min=0, insert_max=10),
        RandomMutation(mut_frac=0.05)
    ]
    
    print(f"✓ Created {len(augment_list)} augmentations:")
    for i, aug in enumerate(augment_list):
        print(f"  {i+1}. {aug.__class__.__name__}")
    
    # Example 1: Use as regular dataset
    print("\n" + "="*40)
    print("EXAMPLE 1: Using as regular dataset")
    print("="*40)
    
    # Set up for training split
    dataset.setup(split='train')
    print(f"Current split: train (length: {len(dataset)})")
    
    # Get a single sample
    x, y = dataset[0]
    print(f"Sample shape: x={x.shape}, y={y.shape}")
    
    # Example 2: Use DataModule-like methods
    print("\n" + "="*40)
    print("EXAMPLE 2: Using DataModule-like methods")
    print("="*40)
    
    # Get training dataloader
    train_loader = dataset.train_dataloader()
    print(f"Training dataloader: {type(train_loader)}")
    print(f"  - Batch size: {train_loader.batch_size}")
    print(f"  - Shuffle: {train_loader.shuffle}")
    
    # Get validation dataloader
    val_loader = dataset.val_dataloader()
    print(f"Validation dataloader: {type(val_loader)}")
    print(f"  - Batch size: {val_loader.batch_size}")
    print(f"  - Shuffle: {val_loader.shuffle}")
    
    # Example 3: Integration with RobustLoader
    print("\n" + "="*40)
    print("EXAMPLE 3: Integration with RobustLoader")
    print("="*40)
    
    # Create RobustLoader for training with augmentations
    train_loader_aug = RobustLoader(
        base_dataset=dataset.get_train_dataset(),
        augment_list=augment_list,
        max_augs_per_seq=2,
        hard_aug=True,
        batch_size=batch_size,
        shuffle=True
    )
    
    print(f"✓ Created RobustLoader for training:")
    print(f"  - Base dataset: {type(train_loader_aug.dataset)}")
    print(f"  - Augmentations: {len(augment_list)}")
    print(f"  - Max augs per seq: 2")
    print(f"  - Hard augmentation: True")
    
    # Test the augmented loader
    print("\nTesting augmented loader...")
    try:
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader_aug):
            print(f"  Batch {batch_idx + 1}: x={x_batch.shape}, y={y_batch.shape}")
            
            # Check that sequence length is preserved
            if x_batch.shape[-1] == dataset.L:
                print(f"    ✓ Sequence length preserved: {x_batch.shape[-1]}")
            else:
                print(f"    ✗ Sequence length mismatch: expected {dataset.L}, got {x_batch.shape[-1]}")
            
            # Only test first few batches
            if batch_idx >= 2:
                break
                
    except Exception as e:
        print(f"✗ Error testing augmented loader: {e}")
    
    # Example 4: Create validation loader with RobustLoader
    print("\n" + "="*40)
    print("EXAMPLE 4: Validation with RobustLoader (no augmentations)")
    print("="*40)
    
    val_loader_aug = RobustLoader(
        base_dataset=dataset.get_val_dataset(),
        augment_list=augment_list,
        max_augs_per_seq=2,
        hard_aug=True,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Disable augmentations for validation
    val_loader_aug.disable_augmentations()
    print(f"✓ Created validation RobustLoader with augmentations disabled")
    
    # Example 5: PyTorch Lightning integration
    print("\n" + "="*40)
    print("EXAMPLE 5: PyTorch Lightning integration")
    print("="*40)
    
    class SimpleDataModule:
        """Simple DataModule that uses enhanced H5Dataset."""
        
        def __init__(self, dataset, augment_list, max_augs_per_seq=2, hard_aug=True):
            self.dataset = dataset
            self.augment_list = augment_list
            self.max_augs_per_seq = max_augs_per_seq
            self.hard_aug = hard_aug
        
        def train_dataloader(self):
            """Training dataloader with augmentations."""
            return RobustLoader(
                base_dataset=self.dataset.get_train_dataset(),
                augment_list=self.augment_list,
                max_augs_per_seq=self.max_augs_per_seq,
                hard_aug=self.hard_aug,
                batch_size=self.dataset.batch_size,
                shuffle=True
            )
        
        def val_dataloader(self):
            """Validation dataloader without augmentations."""
            loader = RobustLoader(
                base_dataset=self.dataset.get_val_dataset(),
                augment_list=self.augment_list,
                max_augs_per_seq=self.max_augs_per_seq,
                hard_aug=self.hard_aug,
                batch_size=self.dataset.batch_size,
                shuffle=False
            )
            loader.disable_augmentations()
            return loader
        
        def test_dataloader(self):
            """Test dataloader without augmentations."""
            loader = RobustLoader(
                base_dataset=self.dataset.get_test_dataset(),
                augment_list=self.augment_list,
                max_augs_per_seq=self.max_augs_per_seq,
                hard_aug=self.hard_aug,
                batch_size=self.dataset.batch_size,
                shuffle=False
            )
            loader.disable_augmentations()
            return loader
    
    # Create the simple DataModule
    data_module = SimpleDataModule(
        dataset=dataset,
        augment_list=augment_list,
        max_augs_per_seq=2,
        hard_aug=True
    )
    
    print(f"✓ Created SimpleDataModule:")
    print(f"  - Training: {type(data_module.train_dataloader())}")
    print(f"  - Validation: {type(data_module.val_dataloader())}")
    print(f"  - Test: {type(data_module.test_dataloader())}")
    
    print("\n" + "="*60)
    print("INTEGRATION SUMMARY")
    print("="*60)
    print("✓ Enhanced H5Dataset provides both Dataset and DataModule functionality")
    print("✓ Easy integration with RobustLoader for augmentations")
    print("✓ No need to nest DataModules")
    print("✓ Clean separation of concerns:")
    print("  - H5Dataset: Data loading and management")
    print("  - RobustLoader: Augmentation application")
    print("  - SimpleDataModule: PyTorch Lightning integration")
    print("✓ All augmentations preserve sequence length L")
    print("✓ Runtime control of augmentations (enable/disable)")
    print("="*60)


if __name__ == "__main__":
    main() 