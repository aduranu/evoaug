#!/usr/bin/env python3
"""
DeepSTARR Training Script with EvoAug2 DataLoader Version

This script converts the Jupyter notebook to use evoaug2 with the dataloader version
instead of the model wrapper approach.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from evoaug.augment import (
    RandomDeletion, RandomRC, RandomInsertion,
    RandomTranslocation, RandomMutation, RandomNoise
)
from evoaug.evoaug import RobustLoader
from utils.model_zoo import DeepSTARRModel, DeepSTARR
from utils import utils
from scipy import stats

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set PyTorch precision for better performance on Tensor Cores
torch.set_float32_matmul_precision('medium')


def main():
    """Main training function implementing the two-stage EvoAug2 approach"""
    
    print("="*70)
    print("DEEPSTARR TRAINING WITH TWO-STAGE EVOAUG2 APPROACH")
    print("="*70)
    print("This script implements the research methodology described in your paper:")
    print("")
    print("STAGE 1: Train DNN with EvoAug augmentations applied stochastically online")
    print("         → Goal: Learn robust representations of features (e.g., motifs)")
    print("         → Method: Expose model to expanded genetic variation")
    print("         → Expected: Preserve motifs on average while learning robustness")
    print("")
    print("STAGE 2: Fine-tune the augmented model on original, unperturbed data")
    print("         → Goal: Remove augmentation bias and refine features")
    print("         → Method: Continue training on original data with lower learning rate")
    print("         → Expected: Guide function towards observed biology")
    print("")
    print("CONTROL: Train separate model on original data only (baseline comparison)")
    print("="*70)
    print("")
    
    # Configuration
    expt_name = 'DeepSTARR'
    data_path = '.'
    filepath = '/grid/koo/home/duran/evoaug/deepstarr-data.h5'
    output_dir = '/grid/koo/home/duran/evoaug/testing/'
    batch_size = 128
    
    # Check if data file exists
    if not os.path.exists(filepath):
        print(f"Data file not found: {filepath}")
        print("Please download the DeepSTARR data first:")
        print("wget https://zenodo.org/record/7265991/files/DeepSTARR_data.h5")
        return
    
    # Define augmentations using optimal DeepSTARR hyperparameters from hyperparameter search
    # Based on Additional file 1: Figs. S1, S3, and S4 from the paper
    augment_list = [
        #RandomDeletion(delete_min=0, delete_max=30),      # DeepSTARR optimal: delete_max = 30
        RandomTranslocation(shift_min=0, shift_max=20),   # DeepSTARR optimal: shift_max = 20
        RandomInsertion(insert_min=0, insert_max=20),     # DeepSTARR optimal: insert_max = 20
        RandomRC(rc_prob=0.0),                           # DeepSTARR optimal: rc_prob = 0 (no reverse-complement)
        RandomMutation(mut_frac=0.05),                    # DeepSTARR optimal: mutate_frac = 0.05
        RandomNoise(noise_mean=0, noise_std=0.3),        # DeepSTARR optimal: noise_std = 0.3
    ]
    
    print("Using DeepSTARR optimal augmentation hyperparameters:")
    print("  - Deletion: delete_min=0, delete_max=30")
    print("  - Translocation: shift_min=0, shift_max=20")
    print("  - Insertion: insert_min=0, insert_max=20")
    print("  - Reverse-complement: rc_prob=0.0 (disabled)")
    print("  - Mutation: mutate_frac=0.05")
    print("  - Noise: noise_mean=0, noise_std=0.3")
    print("  - Max augmentations per sequence: 2")
    print("  - Augmentation setting: Hard (always apply exactly 2 augmentations)")
    print("  - Augmentation priority order: inversion, deletion, translocation, insertion, reverse-complement, mutation, noise")
    
    print("=== Stage 1: Training with EvoAug2 Augmentations ===")
    print("Training DNN on sequences with EvoAug augmentations applied stochastically online...")
    print("Goal: Enhance model's ability to learn robust representations of features (e.g., motifs)")
    print("      by exposing it to expanded genetic variation while preserving motifs on average")
    
    # Create model
    deepstarr = DeepSTARR(2)
    
    # Create Lightning model
    model = DeepSTARRModel(deepstarr, learning_rate=0.001, weight_decay=1e-6)
    
    # Create enhanced H5Dataset (now includes DataModule-like functionality)
    base_dataset = utils.H5Dataset(filepath, batch_size=batch_size, lower_case=False, transpose=False)
    
    # Create augmented data module for Stage 1 training
    class AugmentedDataModule(pl.LightningDataModule):
        def __init__(self, base_dataset, augment_list, max_augs_per_seq, hard_aug):
            super().__init__()
            self.base_dataset = base_dataset
            self.augment_list = augment_list
            self.max_augs_per_seq = max_augs_per_seq
            self.hard_aug = hard_aug
            
        def train_dataloader(self):
            # Use RobustLoader with training dataset
            train_dataset = self.base_dataset.get_train_dataset()
            return RobustLoader(
                base_dataset=train_dataset,
                augment_list=self.augment_list,
                max_augs_per_seq=self.max_augs_per_seq,
                hard_aug=self.hard_aug,
                batch_size=self.base_dataset.batch_size,
                shuffle=True
            )
        
        def val_dataloader(self):
            # Use RobustLoader with validation dataset and disable augmentations
            val_dataset = self.base_dataset.get_val_dataset()
            loader = RobustLoader(
                base_dataset=val_dataset,
                augment_list=self.augment_list,
                max_augs_per_seq=self.max_augs_per_seq,
                hard_aug=self.hard_aug,
                batch_size=self.base_dataset.batch_size,
                shuffle=False
            )
            loader.disable_augmentations()
            return loader
        
        def test_dataloader(self):
            # Use RobustLoader with test dataset and disable augmentations
            test_dataset = self.base_dataset.get_test_dataset()
            loader = RobustLoader(
                base_dataset=test_dataset,
                augment_list=self.augment_list,
                max_augs_per_seq=self.max_augs_per_seq,
                hard_aug=self.hard_aug,
                batch_size=self.base_dataset.batch_size,
                shuffle=False
            )
            loader.disable_augmentations()
            return loader
    
    # Create augmented data module for Stage 1
    data_module = AugmentedDataModule(
        base_dataset, 
        augment_list, 
        max_augs_per_seq=2,  # DeepSTARR optimal: maximum 2 augmentations per sequence
        hard_aug=True         # DeepSTARR uses hard setting: always apply exactly 2 augmentations
    )
    
    # Create trainer
    ckpt_aug_path = expt_name + "_aug"
    callback_topmodel = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        save_top_k=1,
        dirpath=output_dir,
        filename=ckpt_aug_path
    )
    callback_es = pl.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    
    # Create custom trainer that uses our augmented training dataloader
    trainer = pl.Trainer(
        max_epochs=100,
        logger=None,
        callbacks=[callback_es, callback_topmodel],
        accelerator='auto',
        devices='auto',
    )
    
    # Train model with augmentations using Lightning trainer
    print(f"Starting Stage 1 training with EvoAug2 augmentations...")
    print(f"Augmented model will be saved to: {os.path.join(output_dir, ckpt_aug_path + '.ckpt')}")
    print("Training on augmented data to learn robust feature representations...")
    
    # Use Lightning trainer with augmented data module
    trainer.fit(model, datamodule=data_module)
    
    print("=== Stage 1: Training with augmentations completed ===")
    
    # Load best model and evaluate
    best_model_path = os.path.join(output_dir, ckpt_aug_path + ".ckpt")
    if os.path.exists(best_model_path):
        print(f"Loading best model from: {best_model_path}")
        model = DeepSTARRModel.load_from_checkpoint(best_model_path, model=deepstarr)
        
        # Evaluate on test set (no augmentations used during evaluation)
        print("Evaluating augmented model on test set (no augmentations)...")
        trainer.test(model, datamodule=data_module)
        
        # Get predictions
        # Note: For models with insertion augmentation, sequences are padded at 3' end with random DNA
        # This is handled automatically by the RobustLoader during training
        pred = utils.get_predictions(model, base_dataset.x_test, batch_size=batch_size)
        results = utils.evaluate_model(base_dataset.y_test, pred, task='regression')
        
        # Print correlation metrics
        y_true = base_dataset.y_test
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
    
    print("=== Stage 2: Fine-tuning on Original Data ===")
    print("Fine-tuning the augmented model on original, unperturbed data to remove augmentation bias...")
    print("Fine-tuning parameters: 5 epochs, learning rate 0.0001, weight decay 1e-6")
    
    # Load the best augmented model for fine-tuning
    if os.path.exists(best_model_path):
        print(f"Loading best augmented model for fine-tuning from: {best_model_path}")
        # Load the model with the same architecture and weights from augmented training
        model_finetune = DeepSTARRModel.load_from_checkpoint(best_model_path, model=deepstarr)
        # Update learning rate for fine-tuning (lower learning rate for refinement)
        model_finetune.learning_rate = 0.0001
        model_finetune.configure_optimizers()  # Reconfigure with new learning rate
        print("✓ Successfully loaded augmented model for fine-tuning")
    else:
        print("✗ ERROR: No augmented model found for fine-tuning!")
        print("   Fine-tuning requires the augmented model from Stage 1.")
        return
    
    # Create data module without augmentations for fine-tuning
    # Use the same base_dataset but create a simple DataModule for fine-tuning
    class FineTuneDataModule(pl.LightningDataModule):
        def __init__(self, base_dataset):
            super().__init__()
            self.base_dataset = base_dataset
            
        def train_dataloader(self):
            return self.base_dataset.train_dataloader()
            
        def val_dataloader(self):
            return self.base_dataset.val_dataloader()
            
        def test_dataloader(self):
            return self.base_dataset.test_dataloader()
    
    data_module_finetune = FineTuneDataModule(base_dataset)
    
    # Create trainer for fine-tuning
    ckpt_finetune_path = expt_name + "_finetune"
    callback_topmodel_finetune = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        save_top_k=1,
        dirpath=output_dir,
        filename=ckpt_finetune_path
    )
    
    trainer_finetune = pl.Trainer(
        max_epochs=5,  # Fine-tuning for 5 epochs as specified in the paper
        logger=None,
        callbacks=[callback_topmodel_finetune],
        accelerator='auto',
        devices='auto'
    )
    
    # Fine-tune model on original data
    print(f"Starting fine-tuning on original, unperturbed data...")
    print(f"Fine-tuned model will be saved to: {os.path.join(output_dir, ckpt_finetune_path + '.ckpt')}")
    print("Goal: Remove augmentation bias and refine features towards observed biology")
    trainer_finetune.fit(model_finetune, datamodule=data_module_finetune)
    
    print("=== Fine-tuning on original data completed ===")
    
    # Load best fine-tuned model and evaluate
    best_finetune_path = os.path.join(output_dir, ckpt_finetune_path + ".ckpt")
    if os.path.exists(best_finetune_path):
        print(f"Loading best fine-tuned model from: {best_finetune_path}")
        model_finetune = DeepSTARRModel.load_from_checkpoint(best_finetune_path, model=deepstarr)
        
        # Evaluate on test set (no augmentations used during evaluation)
        print("Evaluating fine-tuned model on test set (no augmentations)...")
        trainer_finetune.test(model_finetune, datamodule=data_module_finetune)
        
        # Get predictions
        # Note: Fine-tuned model is evaluated on original, unperturbed sequences
        pred_finetune = utils.get_predictions(model_finetune, base_dataset.x_test, batch_size=batch_size)
        results_finetune = utils.evaluate_model(base_dataset.y_test, pred_finetune, task='regression')
        
        # Print correlation metrics
        y_true_finetune = base_dataset.y_test
        y_score_finetune = pred_finetune
        
        print('Fine-tuned Pearson r')
        vals = []
        for class_index in range(y_true_finetune.shape[-1]):
            vals.append(stats.pearsonr(y_true_finetune[:,class_index], y_score_finetune[:,class_index])[0])
        print(np.array(vals))
        
        print('Fine-tuned Spearman rho')
        vals = []
        for class_index in range(y_true_finetune.shape[-1]):
            vals.append(stats.spearmanr(y_true_finetune[:,class_index], y_score_finetune[:,class_index])[0])
        print(np.array(vals))
    else:
        print("✗ ERROR: Fine-tuned model was not saved!")
    
    print("=== Control: Training on Original Data Only ===")
    print("Training a separate DNN model from scratch on original, unperturbed data...")
    print("Purpose: Baseline comparison to evaluate the effectiveness of the two-stage approach")
    
    # Create control model (no augmentations)
    deepstarr_control = DeepSTARR(2)
    model_control = DeepSTARRModel(deepstarr_control, learning_rate=0.001, weight_decay=1e-6)
    
    # Use the same base_dataset for control training
    data_module_control = FineTuneDataModule(base_dataset)
    
    # Create trainer for control
    ckpt_control_path = expt_name + "_standard"
    callback_topmodel_control = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        save_top_k=1,
        dirpath=output_dir,
        filename=ckpt_control_path
    )
    callback_es_control = pl.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    
    trainer_control = pl.Trainer(
        max_epochs=100,
        logger=None,
        callbacks=[callback_es_control, callback_topmodel_control],
        accelerator='auto',
        devices='auto'
    )
    
    # Train control model on original data
    print(f"Starting control training on original, unperturbed data...")
    print(f"Control model will be saved to: {os.path.join(output_dir, ckpt_control_path + '.ckpt')}")
    print("Training from scratch without any augmentations for baseline comparison...")
    trainer_control.fit(model_control, datamodule=data_module_control)
    
    print("=== Control training on original data completed ===")
    
    # Load best control model and evaluate
    best_control_path = os.path.join(output_dir, ckpt_control_path + ".ckpt")
    if os.path.exists(best_control_path):
        print(f"Loading best control model from: {best_control_path}")
        model_control = DeepSTARRModel.load_from_checkpoint(best_control_path, model=deepstarr_control)
        
        # Evaluate on test set (no augmentations used during evaluation)
        print("Evaluating control model on test set (no augmentations)...")
        trainer_control.test(model_control, datamodule=data_module_control)
        
        # Get predictions
        # Note: Control model is evaluated on original, unperturbed sequences
        pred_control = utils.get_predictions(model_control, base_dataset.x_test, batch_size=batch_size)
        results_control = utils.evaluate_model(base_dataset.y_test, pred_control, task='regression')
        
        # Print correlation metrics
        y_true_control = base_dataset.y_test
        y_score_control = pred_control
        
        print('Control Pearson r')
        vals = []
        for class_index in range(y_true_control.shape[-1]):
            vals.append(stats.pearsonr(y_true_control[:,class_index], y_score_control[:,class_index])[0])
        print(np.array(vals))
        
        print('Control Spearman rho')
        vals = []
        for class_index in range(y_true_control.shape[-1]):
            vals.append(stats.spearmanr(y_true_control[:,class_index], y_score_control[:,class_index])[0])
        print(np.array(vals))
    
    print("=== All training completed ===")
    
    # Summary of saved models
    print("\n" + "="*50)
    print("TWO-STAGE TRAINING SUMMARY")
    print("="*50)
    
    saved_models = []
    
    # Check Stage 1: Augmented model
    if os.path.exists(os.path.join(output_dir, ckpt_aug_path + ".ckpt")):
        saved_models.append(f"✓ {ckpt_aug_path}.ckpt - Stage 1: Model trained with EvoAug2 augmentations")
        saved_models.append("   → Learned robust representations from expanded genetic variation")
    else:
        saved_models.append(f"✗ {ckpt_aug_path}.ckpt - Stage 1: NOT SAVED")
    
    # Check Stage 2: Fine-tuned model
    if os.path.exists(os.path.join(output_dir, ckpt_finetune_path + ".ckpt")):
        saved_models.append(f"✓ {ckpt_finetune_path}.ckpt - Stage 2: Model fine-tuned on original data")
        saved_models.append("   → Removed augmentation bias, refined features towards observed biology")
    else:
        saved_models.append(f"✗ {ckpt_finetune_path}.ckpt - Stage 2: NOT SAVED")
    
    # Check Control model
    if os.path.exists(os.path.join(output_dir, ckpt_control_path + ".ckpt")):
        saved_models.append(f"✓ {ckpt_control_path}.ckpt - Control: Model trained on original data only")
        saved_models.append("   → Baseline comparison for evaluating two-stage approach")
    else:
        saved_models.append(f"✗ {ckpt_control_path}.ckpt - Control: NOT SAVED")
    
    print("\nTraining Stages Completed:")
    for model in saved_models:
        print(f"  {model}")
    
    print(f"\nAll models saved to directory: {os.path.abspath(output_dir)}")
    print("\nTwo-Stage Training Logic:")
    print("  1. Stage 1: Train with augmentations → Learn robust feature representations")
    print("  2. Stage 2: Fine-tune on original data → Remove bias, refine towards biology")
    print("  3. Control: Train on original data only → Baseline comparison")
    print("="*50)


if __name__ == "__main__":
    main() 