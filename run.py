#!/usr/bin/env python3
"""
DeepSTARR Training with EvoAug2 DataLoader Implementation

This script converts the original Jupyter notebook to use the new evoaug2
DataLoader implementation instead of the RobustLoader Lightning module.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy import stats
import h5py
import sys

# Add the current directory to path to import evoaug2
sys.path.append('.')

# Import the new evoaug2 implementation
from evoaug.evoaug2 import RobustLoader

from evoaug.augment import (
    RandomDeletion, RandomRC, RandomInsertion,
    RandomTranslocation, RandomMutation, RandomNoise
)

from utils.model_zoo import DeepSTARR
from utils.utils import configure_optimizer, get_predictions, evaluate_model, H5Dataset

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# Main train loop
def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, 
                num_epochs=100, early_stopping_patience=10, save_path=None):
    """Train model with early stopping and checkpointing."""
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_idx, (sequences, targets) in enumerate(train_dataloader):
            sequences, targets = sequences.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Train Loss: {loss.item():.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, targets in val_dataloader:
                sequences, targets = sequences.to(device), targets.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_dataloader)
        avg_val_loss = val_loss / len(val_dataloader)
        
        print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step(avg_val_loss)
        
        # Early stopping and checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model
            if save_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                }, save_path)
                print(f"Saved best model to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
    
    return model


def main():
    """Main training function."""
    
    # Configuration
    expt_name = 'DeepSTARR'
    data_path = '.'
    filepath = '/grid/koo/home/duran/evoaug/deepstarr-data.h5'
    output_dir = '.'
    batch_size = 128
    
    print("Loading datasets...")
    
    # Load datasets
    train_dataset = H5Dataset(filepath, split='train')
    val_dataset = H5Dataset(filepath, split='val')
    test_dataset = H5Dataset(filepath, split='test')
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Define augmentations according to DeepSTARR paper specifications
    augment_list = [
        RandomDeletion(delete_min=0, delete_max=30),
        RandomTranslocation(shift_min=0, shift_max=20),
        RandomInsertion(insert_min=0, insert_max=0),
        RandomRC(rc_prob=0),  # Set to 0 as per DeepSTARR specifications
        RandomMutation(mut_frac=0.05),
        RandomNoise(noise_mean=0, noise_std=0.3),
    ]
    
    print("Creating dataloaders...")
    
    # Create augmented dataloader for training
    # DeepSTARR uses max_augs_per_seq=2 with hard_aug=True (always apply exactly 2 augmentations)

    train_dataloader = RobustLoader(
        base_dataset=train_dataset,
        augment_list=augment_list,
        max_augs_per_seq=2,
        hard_aug=True,
        batch_size=batch_size,
    )
    
    # Create regular dataloader for validation
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create validation dataloader
    val_dataloader = RobustLoader(
        base_dataset=val_dataset,
        augment_list=augment_list,
        max_augs_per_seq=2,
        hard_aug=True,
        batch_size=batch_size,
    )
    # no augmentations (like test set on paper?)
    val_dataloader.disable_augmentations()
    
    # Create model and training components
    model = DeepSTARR(2).to(device)
    criterion = nn.MSELoss()
    # DeepSTARR training parameters: lr=0.001, weight_decay=1e-6, decay_factor=0.1, patience=5
    optimizer_dict = configure_optimizer(model, lr=0.001, weight_decay=1e-6, 
                                             decay_factor=0.1, patience=5, monitor='val_loss')

    optimizer = optimizer_dict['optimizer']
    lr_scheduler = optimizer_dict['lr_scheduler']
    scheduler = lr_scheduler['scheduler']
    # monitor = lr_scheduler['monitor']
    
    print("Starting training with augmentations...")
    
    # Train with augmentations
    # DeepSTARR training: 100 epochs, early stopping patience=10
    # ckpt_aug_path = os.path.join(output_dir, expt_name+"_aug.pt")
    # model = train_model(
    #     model=model,
    #     train_dataloader=train_dataloader,
    #     val_dataloader=val_dataloader,
    #     criterion=criterion,
    #     optimizer=optimizer,
    #     scheduler=scheduler,
    #     num_epochs=100,  # DeepSTARR trains for 100 epochs
    #     early_stopping_patience=10,  # Early stopping patience of 10 epochs
    #     save_path=ckpt_aug_path
    # )

    ### continue here
    
    # Load best model
    ckpt_aug_path = '/grid/koo/home/duran/evoaug/DeepSTARR_aug.pt'
    checkpoint = torch.load(ckpt_aug_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']} with val_loss: {checkpoint['val_loss']:.4f}")
    
    # Evaluate best model
    print("Evaluating best model (from first augmentation)...")
    pred = get_predictions(model, test_dataset.x, batch_size=batch_size)
    results = evaluate_model(test_dataset.y, pred, task='regression')
    
    # Print correlation metrics
    y_true = test_dataset.y
    y_score = pred
    
    print('Pearson r')
    vals = []
    for class_index in range(y_true.shape[-1]):
        vals.append(stats.pearsonr(y_true[:,class_index], y_score[:,class_index])[0])
    print(np.array(vals))
    
    print('Spearman rho')
    vals = []
    for class_index in range(y_true.shape[-1]):
        vals.append(stats.spearmanr(y_true[:,class_index], y_score[:,class_index])[0])
    print(np.array(vals))
    
    # Fine-tuning without augmentations
    print("\nStarting fine-tuning without augmentations...")
    
    # Create finetune dataloader (no augmentations)
    finetune_dataloader = RobustLoader(
        base_dataset=train_dataset,
        augment_list=augment_list,
        max_augs_per_seq=2,
        hard_aug=True,
        batch_size=batch_size,
    )
    finetune_dataloader.disable_augmentations()
    
    
    # Create new optimizer for fine-tuning
    # DeepSTARR fine-tuning: lr=0.0001, weight_decay=1e-6, 5 epochs
    finetune_optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.0001, weight_decay=1e-6
    )
    
    # # Fine-tune model
    # ckpt_finetune_path = os.path.join(output_dir, expt_name+"_finetune.pt")
    # model = train_model(
    #     model=model,
    #     train_dataloader=finetune_dataloader,
    #     val_dataloader=val_dataloader,
    #     criterion=criterion,
    #     optimizer=finetune_optimizer,
    #     scheduler=None,  # No scheduler for fine-tuning
    #     num_epochs=5,  # DeepSTARR fine-tunes for 5 epochs
    #     early_stopping_patience=5,
    #     save_path=ckpt_finetune_path
    # )

    # Load best fine-tuned model
    checkpoint = torch.load(ckpt_finetune_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best fine-tuned model from epoch {checkpoint['epoch']} with val_loss: {checkpoint['val_loss']:.4f}")
    
    # Evaluate fine-tuned model
    print("Evaluating fine-tuned model...")
    pred = get_predictions(model, test_dataset.x, batch_size=batch_size)
    results = evaluate_model(test_dataset.y, pred, task='regression')
    
    # Print correlation metrics for fine-tuned model
    y_true = test_dataset.y
    y_score = pred
    
    print('Fine-tuned Pearson r')
    vals = []
    for class_index in range(y_true.shape[-1]):
        vals.append(stats.pearsonr(y_true[:,class_index], y_score[:,class_index])[0])
    print(np.array(vals))
    
    print('Fine-tuned Spearman rho')
    vals = []
    for class_index in range(y_true.shape[-1]):
        vals.append(stats.spearmanr(y_true[:,class_index], y_score[:,class_index])[0])
    print(np.array(vals))

    ## NOTE: continue here
    
    # Control: train model with standard training (no augmentations)
    print("\nTraining control model (no augmentations)...")
    
    # Create standard dataloader (no augmentations)
    standard_train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Create new model for control
    control_model = DeepSTARR(2).to(device)
    # Control model uses same training parameters as augmented model
    control_optimizer_dict = configure_optimizer(
        control_model, lr=0.001, weight_decay=1e-6, 
        decay_factor=0.1, patience=5, monitor='val_loss'
    )

    control_optimizer = control_optimizer_dict['optimizer']
    control_lr_scheduler = control_optimizer_dict['lr_scheduler']
    control_scheduler = control_lr_scheduler['scheduler']
    # monitor = control_lr_scheduler['monitor']
    
    # Train control model
    ckpt_standard_path = os.path.join(output_dir, expt_name+"_standard.pt")
    control_model = train_model(
        model=control_model,
        train_dataloader=standard_train_dataloader,
        val_dataloader=val_dataloader,
        criterion=criterion,
        optimizer=control_optimizer,
        scheduler=control_scheduler,
        num_epochs=100,  # Same as augmented training
        early_stopping_patience=10,  # Same as augmented training
        save_path=ckpt_standard_path
    )
    
    # Load best control model
    checkpoint = torch.load(ckpt_standard_path)
    control_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best control model from epoch {checkpoint['epoch']} with val_loss: {checkpoint['val_loss']:.4f}")
    
    # Evaluate control model
    print("Evaluating control model...")
    pred = get_predictions(control_model, test_dataset.x, batch_size=batch_size)
    results = evaluate_model(test_dataset.y, pred, task='regression')
    
    # Print correlation metrics for control model
    y_true = test_dataset.y
    y_score = pred
    
    print('Control Pearson r')
    vals = []
    for class_index in range(y_true.shape[-1]):
        vals.append(stats.pearsonr(y_true[:,class_index], y_score[:,class_index])[0])
    print(np.array(vals))
    
    print('Control Spearman rho')
    vals = []
    for class_index in range(y_true.shape[-1]):
        vals.append(stats.spearmanr(y_true[:,class_index], y_score[:,class_index])[0])
    print(np.array(vals))
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main() 