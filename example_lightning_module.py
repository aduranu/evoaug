#!/usr/bin/env python3
"""
EvoAug2 PyTorch Lightning Example

Complete training script with Lightning integration and two-stage approach.
Shows DataModule creation, checkpoint management, and performance comparison.

This script demonstrates the complete EvoAug2 workflow including:

1. **Stage 1**: Training with evolution-inspired augmentations
2. **Stage 2**: Fine-tuning on original data to remove augmentation bias
3. **Control**: Standard training without augmentations for comparison
4. **Evaluation**: Comprehensive performance analysis and visualization

Usage
-----
    python example_lightning_module.py

Prerequisites
------------
- PyTorch >= 1.9.0
- PyTorch Lightning >= 1.5.0
- DeepSTARR dataset (automatically downloaded if not present)
- GPU recommended for training

Outputs
-------
- Trained model checkpoints (.ckpt files)
- Performance comparison plots (.png files)
- Training logs and evaluation metrics

See Also
--------
- :doc:`examples/lightning_module` for detailed documentation
- :doc:`quickstart` for basic usage instructions
- :doc:`user_guide/training` for training strategies
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
from evoaug_utils.model_zoo import DeepSTARRModel, DeepSTARR
from evoaug_utils import utils
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_float32_matmul_precision('medium')
plt.style.use('default')
sns.set_palette("husl")

def check_existing_checkpoints(output_dir, expt_name):
    """Check for existing checkpoints and return their status.
    
    This function scans the output directory for existing model checkpoints
    and returns information about their availability and training progress.
    
    Parameters
    ----------
    output_dir : str
        Directory path where checkpoints are stored
    expt_name : str
        Base name for the experiment (e.g., 'DeepSTARR')
    
    Returns
    -------
    dict
        Dictionary containing checkpoint status for each training stage:
        - 'augmented': Stage 1 model checkpoint status
        - 'finetuned': Stage 2 model checkpoint status  
        - 'control': Control model checkpoint status
        
        Each stage contains:
        - 'exists': Boolean indicating if checkpoint exists
        - 'path': Full path to checkpoint file (if exists)
        - 'epochs': Number of training epochs completed (if available)
    
    Examples
    --------
    >>> status = check_existing_checkpoints('/path/to/output', 'DeepSTARR')
    >>> if status['augmented']['exists']:
    ...     print(f"Found augmented model: {status['augmented']['path']}")
    """
    checkpoint_status = {
        'augmented': {'exists': False, 'path': None, 'epochs': None},
        'finetuned': {'exists': False, 'path': None, 'epochs': None},
        'control': {'exists': False, 'path': None, 'epochs': None}
    }
    
    # Check for augmented model checkpoint
    aug_path = os.path.join(output_dir, f"{expt_name}_aug.ckpt")
    if os.path.exists(aug_path):
        checkpoint_status['augmented']['exists'] = True
        checkpoint_status['augmented']['path'] = aug_path
        try:
            checkpoint = torch.load(aug_path, map_location='cpu')
            if 'epoch' in checkpoint:
                checkpoint_status['augmented']['epochs'] = checkpoint['epoch']
        except:
            pass
    
    # Check for fine-tuned model checkpoint
    finetune_path = os.path.join(output_dir, f"{expt_name}_finetune.ckpt")
    if os.path.exists(finetune_path):
        checkpoint_status['finetuned']['exists'] = True
        checkpoint_status['finetuned']['path'] = finetune_path
        try:
            checkpoint = torch.load(finetune_path, map_location='cpu')
            if 'epoch' in checkpoint:
                checkpoint_status['finetuned']['epochs'] = checkpoint['epoch']
        except:
            pass
    
    # Check for control model checkpoint
    control_path = os.path.join(output_dir, f"{expt_name}_standard.ckpt")
    if os.path.exists(control_path):
        checkpoint_status['control']['exists'] = True
        checkpoint_status['control']['path'] = control_path
        try:
            checkpoint = torch.load(control_path, map_location='cpu')
            if 'epoch' in checkpoint:
                checkpoint_status['control']['epochs'] = checkpoint['epoch']
        except:
            pass
    
    return checkpoint_status

def print_checkpoint_status(checkpoint_status):
    """Print the status of existing checkpoints.
    
    Displays a formatted summary of checkpoint availability and training progress
    for each training stage.
    
    Parameters
    ----------
    checkpoint_status : dict
        Dictionary returned by :func:`check_existing_checkpoints`
    
    Examples
    --------
    >>> status = check_existing_checkpoints('/path/to/output', 'DeepSTARR')
    >>> print_checkpoint_status(status)
    ==================================================
    EXISTING CHECKPOINT STATUS
    ==================================================
    ✓ Augmented model: DeepSTARR_aug.ckpt (epochs: 100)
    ✗ Finetuned model: NOT FOUND
    ✓ Control model: DeepSTARR_standard.ckpt (epochs: 95)
    ==================================================
    """
    print("\n" + "="*50)
    print("EXISTING CHECKPOINT STATUS")
    print("="*50)
    
    for stage, status in checkpoint_status.items():
        if status['exists']:
            epoch_info = f" (epochs: {status['epochs']})" if status['epochs'] is not None else ""
            print(f"✓ {stage.capitalize()} model: {os.path.basename(status['path'])}{epoch_info}")
        else:
            print(f"✗ {stage.capitalize()} model: NOT FOUND")
    
    print("="*50)

def create_plots_directory():
    """Create plots directory if it doesn't exist.
    
    Creates a 'plots' directory in the current working directory
    for storing generated visualization files.
    
    Returns
    -------
    str
        Path to the created plots directory
    
    Examples
    --------
    >>> plots_dir = create_plots_directory()
    Created plots directory: plots
    >>> print(plots_dir)
    plots
    """
    plots_dir = "plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        print(f"Created plots directory: {plots_dir}")
    return plots_dir

def plot_metrics_comparison(metrics_data, plots_dir, expt_name):
    """Create comprehensive plots comparing metrics across the three model types."""
    print(f"\nCreating comparison plots in {plots_dir}/...")
    
    # Set up the plotting style
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    
    # 1. Correlation Metrics Comparison (Bar Plot)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{expt_name}: Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # Extract metrics for plotting
    model_names = list(metrics_data.keys())
    pearson_r = [metrics_data[model]['pearson_r'] for model in model_names]
    spearman_r = [metrics_data[model]['spearman_r'] for model in model_names]
    
    # Convert to numpy arrays for easier handling
    pearson_r = np.array(pearson_r)
    spearman_r = np.array(spearman_r)
    
    # Plot Pearson correlation
    ax1 = axes[0, 0]
    x_pos = np.arange(len(model_names))
    bars1 = ax1.bar(x_pos, pearson_r.mean(axis=1), yerr=pearson_r.std(axis=1), 
                     capsize=5, alpha=0.7, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax1.set_xlabel('Model Type')
    ax1.set_ylabel('Pearson Correlation (r)')
    ax1.set_title('Pearson Correlation Comparison')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([name.replace('_', ' ').title() for name in model_names])
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{pearson_r.mean(axis=1)[i]:.3f}', ha='center', va='bottom')
    
    # Plot Spearman correlation
    ax2 = axes[0, 1]
    bars2 = ax2.bar(x_pos, spearman_r.mean(axis=1), yerr=spearman_r.std(axis=1), 
                     capsize=5, alpha=0.7, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax2.set_xlabel('Model Type')
    ax2.set_ylabel('Spearman Correlation (ρ)')
    ax2.set_title('Spearman Correlation Comparison')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([name.replace('_', ' ').title() for name in model_names])
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{spearman_r.mean(axis=1)[i]:.3f}', ha='center', va='bottom')
    
    # 2. Detailed Metrics by Class (Heatmap)
    ax3 = axes[1, 0]
    # Create heatmap data
    heatmap_data = np.zeros((len(model_names), pearson_r.shape[1]))
    for i, model in enumerate(model_names):
        heatmap_data[i, :] = pearson_r[i, :]
    
    im = ax3.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto')
    ax3.set_xlabel('Output Class')
    ax3.set_ylabel('Model Type')
    ax3.set_title('Pearson Correlation by Class')
    ax3.set_xticks(range(pearson_r.shape[1]))
    ax3.set_xticklabels([f'Class {i+1}' for i in range(pearson_r.shape[1])])
    ax3.set_yticks(range(len(model_names)))
    ax3.set_yticklabels([name.replace('_', ' ').title() for name in model_names])
    
    # Add text annotations
    for i in range(len(model_names)):
        for j in range(pearson_r.shape[1]):
            text = ax3.text(j, i, f'{heatmap_data[i, j]:.3f}',
                           ha="center", va="center", color="black", fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('Pearson Correlation')
    
    # 3. Performance Improvement Analysis
    ax4 = axes[1, 1]
    
    # Calculate improvement over control
    if 'control' in metrics_data:
        control_pearson = pearson_r[model_names.index('control')]
        improvements = []
        improvement_labels = []
        
        for i, model in enumerate(model_names):
            if model != 'control':
                improvement = pearson_r[i] - control_pearson
                improvements.append(improvement)
                improvement_labels.append(f'{model.replace("_", " ").title()} vs Control')
        
        if improvements:
            improvements = np.array(improvements)
            x_pos_imp = np.arange(len(improvements))
            bars_imp = ax4.bar(x_pos_imp, improvements.mean(axis=1), 
                              yerr=improvements.std(axis=1), capsize=5, alpha=0.7,
                              color=['#FF6B6B', '#4ECDC4'])
            ax4.set_xlabel('Model Comparison')
            ax4.set_ylabel('Improvement in Pearson Correlation')
            ax4.set_title('Performance Improvement Over Control')
            ax4.set_xticks(x_pos_imp)
            ax4.set_xticklabels(improvement_labels, rotation=45, ha='right')
            ax4.grid(True, alpha=0.3)
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            # Add value labels
            for i, bar in enumerate(bars_imp):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                        f'{improvements.mean(axis=1)[i]:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{expt_name}_metrics_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Individual Model Performance (Detailed View)
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    fig2.suptitle(f'{expt_name}: Individual Model Performance', fontsize=16, fontweight='bold')
    
    for i, model in enumerate(model_names):
        ax = axes2[i]
        
        # Create bar plot for each model
        x_pos = np.arange(pearson_r.shape[1])
        width = 0.35
        
        bars1 = ax.bar(x_pos - width/2, pearson_r[i], width, label='Pearson r', 
                       alpha=0.7, color='#FF6B6B')
        bars2 = ax.bar(x_pos + width/2, spearman_r[i], width, label='Spearman ρ', 
                       alpha=0.7, color='#4ECDC4')
        
        ax.set_xlabel('Output Class')
        ax.set_ylabel('Correlation Coefficient')
        ax.set_title(f'{model.replace("_", " ").title()} Model')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'Class {j+1}' for j in range(pearson_r.shape[1])])
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Add value labels
        for j, bar in enumerate(bars1):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{pearson_r[i, j]:.3f}', ha='center', va='bottom', fontsize=8)
        
        for j, bar in enumerate(bars2):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{spearman_r[i, j]:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{expt_name}_individual_performance.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Summary Statistics Table
    fig3, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    for model in model_names:
        pearson_mean = f"{pearson_r[model_names.index(model)].mean():.4f}"
        pearson_std = f"±{pearson_r[model_names.index(model)].std():.4f}"
        spearman_mean = f"{spearman_r[model_names.index(model)].mean():.4f}"
        spearman_std = f"±{spearman_r[model_names.index(model)].std():.4f}"
        
        table_data.append([
            model.replace('_', ' ').title(),
            f"{pearson_mean} {pearson_std}",
            f"{spearman_mean} {spearman_std}"
        ])
    
    table = ax.table(cellText=table_data,
                     colLabels=['Model Type', 'Pearson r (mean ± std)', 'Spearman ρ (mean ± std)'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.3, 0.35, 0.35])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(table_data) + 1):
        for j in range(3):
            if i == 0:  # Header row
                table[(i, j)].set_facecolor('#4ECDC4')
                table[(i, j)].set_text_props(weight='bold', color='white')
            else:  # Data rows
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#F7F7F7')
                else:
                    table[(i, j)].set_facecolor('#FFFFFF')
    
    plt.title(f'{expt_name}: Performance Summary', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(os.path.join(plots_dir, f'{expt_name}_performance_summary.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved comparison plots to {plots_dir}/")
    print(f"  - {expt_name}_metrics_comparison.png")
    print(f"  - {expt_name}_individual_performance.png") 
    print(f"  - {expt_name}_performance_summary.png")

def main():
    """Main training function implementing the two-stage EvoAug2 approach.
    
    This function orchestrates the complete EvoAug2 training workflow:
    
    1. **Stage 1 - Augmentation Training**: Trains a model using evolution-inspired
       data augmentations to learn robust feature representations
    2. **Stage 2 - Fine-tuning**: Fine-tunes the augmented model on original,
       unperturbed data to remove augmentation bias
    3. **Control Training**: Trains a baseline model without augmentations
       for performance comparison
    4. **Evaluation**: Compares performance across all three approaches
       and generates comprehensive visualizations
    
    The function automatically handles checkpoint management, allowing training
    to resume from any stage if interrupted.
    
    Configuration
    -------------
    The function uses the following default configuration (modifiable):
    
    - **Experiment**: DeepSTARR regulatory sequence prediction
    - **Augmentations**: Translocation, mutation, noise (optimal parameters)
    - **Training**: 100 epochs Stage 1, 5 epochs Stage 2
    - **Batch Size**: 128
    - **Learning Rate**: 0.001 (Stage 1), 0.0001 (Stage 2)
    
    Outputs
    --------
    - Model checkpoints for each training stage
    - Performance comparison plots and metrics
    - Training logs and evaluation results
    
    See Also
    --------
    - :doc:`user_guide/training` for detailed training explanations
    - :doc:`examples/lightning_module` for usage examples
    - :doc:`api/evoaug` for API reference
    """
    
    # Configuration
    expt_name = 'DeepSTARR'
    data_path = '.'
    filepath = '/grid/koo/home/duran/evoaug/deepstarr-data.h5'
    output_dir = '/grid/koo/home/duran/evoaug/testing/'
    batch_size = 128
    
    # Create plots directory
    plots_dir = create_plots_directory()
    
    # Check if data file exists
    if not os.path.exists(filepath):
        print(f"Data file not found: {filepath}")
        print("Please download the DeepSTARR data first:")
        print("wget https://zenodo.org/record/7265991/files/DeepSTARR_data.h5")
        return
    
    # Check for existing checkpoints
    checkpoint_status = check_existing_checkpoints(output_dir, expt_name)
    print_checkpoint_status(checkpoint_status)
    
    # Define augmentations using optimal DeepSTARR hyperparameters from hyperparameter search
    # Based on Additional file 1: Figs. S1, S3, and S4 from the paper
    augment_list = [
        #RandomDeletion(delete_min=0, delete_max=30),      # DeepSTARR optimal: delete_max = 30
        RandomTranslocation(shift_min=0, shift_max=20),   # DeepSTARR optimal: shift_max = 20
        # RandomInsertion(insert_min=0, insert_max=20),     # DeepSTARR optimal: insert_max = 20
        RandomRC(rc_prob=0.0),                           # DeepSTARR optimal: rc_prob = 0 (no reverse-complement)
        RandomMutation(mut_frac=0.05),                    # DeepSTARR optimal: mutate_frac = 0.05
        RandomNoise(noise_mean=0, noise_std=0.3),        # DeepSTARR optimal: noise_std = 0.3
    ]
    
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
    
    # Initialize metrics storage
    metrics_data = {}
    
    # STAGE 1: Training with EvoAug2 Augmentations
    print("\n=== Stage 1: Training with EvoAug2 Augmentations ===")
    
    if checkpoint_status['augmented']['exists']:
        print(f"✓ Found existing augmented model checkpoint: {os.path.basename(checkpoint_status['augmented']['path'])}")
        print("Skipping Stage 1 training - using existing model.")
        best_model_path = checkpoint_status['augmented']['path']
    else:
        print("Training DNN on sequences with EvoAug augmentations applied stochastically online...")
        print("Goal: Enhance model's ability to learn robust representations of features (e.g., motifs)")
        print("      by exposing it to expanded genetic variation while preserving motifs on average")
        
        # Create model
        deepstarr = DeepSTARR(2)
        
        # Create Lightning model
        model = DeepSTARRModel(deepstarr, learning_rate=0.001, weight_decay=1e-6)
        
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
        
        # Get the best model path
        best_model_path = os.path.join(output_dir, ckpt_aug_path + ".ckpt")
    
    # Evaluate the augmented model (either newly trained or loaded from checkpoint)
    if os.path.exists(best_model_path):
        print(f"Loading augmented model from: {best_model_path}")
        
        # Create model architecture
        deepstarr = DeepSTARR(2)
        
        # Load the model from checkpoint
        model = DeepSTARRModel.load_from_checkpoint(best_model_path, model=deepstarr)
        
        # Evaluate on test set (no augmentations used during evaluation)
        print("Evaluating augmented model on test set (no augmentations)...")
        
        # Create a simple trainer for evaluation
        eval_trainer = pl.Trainer(
            logger=None,
            accelerator='auto',
            devices='auto',
        )
        
        # Evaluate using the data module
        eval_trainer.test(model, datamodule=data_module)
        
        # Get predictions
        # Note: For models with insertion augmentation, sequences are padded at 3' end with random DNA
        # This is handled automatically by the RobustLoader during training
        pred = utils.get_predictions(model, base_dataset.x_test, batch_size=batch_size)
        results = utils.evaluate_model(base_dataset.y_test, pred, task='regression')
        
        # Print correlation metrics
        y_true = base_dataset.y_test
        y_score = pred
        
        if y_true is not None and y_score is not None:
            print('Pearson r')
            vals = []
            for class_index in range(y_true.shape[-1]):
                vals.append(stats.pearsonr(y_true[:,class_index], y_score[:,class_index])[0])
            pearson_aug = np.array(vals)
            print(pearson_aug)
            
            print('Spearman rho')
            vals = []
            for class_index in range(y_true.shape[-1]):
                vals.append(stats.pearsonr(y_true[:,class_index], y_score[:,class_index])[0])
            spearman_r = np.array(vals)
            print(spearman_r)
            
            # Store metrics for plotting
            metrics_data['augmented'] = {
                'pearson_r': pearson_aug,
                'spearman_r': spearman_r
            }
        else:
            print("Warning: Could not compute metrics - y_true or y_score is None")
            pearson_aug = np.array([])
            spearman_r = np.array([])
    
    # STAGE 2: Fine-tuning on Original Data
    print("\n=== Stage 2: Fine-tuning on Original Data ===")
    
    if checkpoint_status['finetuned']['exists']:
        print(f"✓ Found existing fine-tuned model checkpoint: {os.path.basename(checkpoint_status['finetuned']['path'])}")
        print("Skipping Stage 2 fine-tuning - using existing model.")
        best_finetune_path = checkpoint_status['finetuned']['path']
    else:
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
        
        # Get the best fine-tuned model path
        best_finetune_path = os.path.join(output_dir, ckpt_finetune_path + ".ckpt")
    
    # Evaluate the fine-tuned model
    if os.path.exists(best_finetune_path):
        print(f"Loading best fine-tuned model from: {best_finetune_path}")
        model_finetune = DeepSTARRModel.load_from_checkpoint(best_finetune_path, model=deepstarr)
        
        # Evaluate on test set (no augmentations used during evaluation)
        print("Evaluating fine-tuned model on test set (no augmentations)...")
        
        # Create a simple trainer for evaluation
        eval_trainer = pl.Trainer(
            logger=None,
            accelerator='auto',
            devices='auto'
        )
        
        # Evaluate using the fine-tune data module
        eval_trainer.test(model_finetune, datamodule=data_module_finetune)
        
        # Get predictions
        # Note: Fine-tuned model is evaluated on original, unperturbed sequences
        pred_finetune = utils.get_predictions(model_finetune, base_dataset.x_test, batch_size=batch_size)
        results_finetune = utils.evaluate_model(base_dataset.y_test, pred_finetune, task='regression')
        
        # Print correlation metrics
        y_true_finetune = base_dataset.y_test
        y_score_finetune = pred_finetune
        
        if y_true_finetune is not None and y_score_finetune is not None:
            print('Fine-tuned Pearson r')
            vals = []
            for class_index in range(y_true_finetune.shape[-1]):
                vals.append(stats.pearsonr(y_true_finetune[:,class_index], y_score_finetune[:,class_index])[0])
            pearson_finetune = np.array(vals)
            print(pearson_finetune)
            
            print('Fine-tuned Spearman rho')
            vals = []
            for class_index in range(y_true_finetune.shape[-1]):
                vals.append(stats.spearmanr(y_true_finetune[:,class_index], y_score_finetune[:,class_index])[0])
            spearman_finetune = np.array(vals)
            print(pearson_finetune)
            
            # Store metrics for plotting
            metrics_data['finetuned'] = {
                'pearson_r': pearson_finetune,
                'spearman_r': spearman_finetune
            }
        else:
            print("Warning: Could not compute fine-tuned metrics - y_true or y_score is None")
            pearson_finetune = np.array([])
            spearman_finetune = np.array([])
    else:
        print("✗ ERROR: Fine-tuned model was not saved!")
    
    # CONTROL: Training on Original Data Only
    print("\n=== Control: Training on Original Data Only ===")
    
    if checkpoint_status['control']['exists']:
        print(f"✓ Found existing control model checkpoint: {os.path.basename(checkpoint_status['control']['path'])}")
        print("Skipping Control training - using existing model.")
        best_control_path = checkpoint_status['control']['path']
    else:
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
        
        # Get the best control model path
        best_control_path = os.path.join(output_dir, ckpt_control_path + ".ckpt")
    
    # Evaluate the control model
    if os.path.exists(best_control_path):
        print(f"Loading best control model from: {best_control_path}")
        model_control = DeepSTARRModel.load_from_checkpoint(best_control_path, model=deepstarr_control)
        
        # Evaluate on test set (no augmentations used during evaluation)
        print("Evaluating control model on test set (no augmentations)...")
        
        # Create a simple trainer for evaluation
        eval_trainer = pl.Trainer(
            logger=None,
            accelerator='auto',
            devices='auto'
        )
        
        # Evaluate using the control data module
        eval_trainer.test(model_control, datamodule=data_module_control)
        
        # Get predictions
        # Note: Control model is evaluated on original, unperturbed sequences
        pred_control = utils.get_predictions(model_control, base_dataset.x_test, batch_size=batch_size)
        results_control = utils.evaluate_model(base_dataset.y_test, pred_control, task='regression')
        
        # Print correlation metrics
        y_true_control = base_dataset.y_test
        y_score_control = pred_control
        
        if y_true_control is not None and y_score_control is not None:
            print('Control Pearson r')
            vals = []
            for class_index in range(y_true_control.shape[-1]):
                vals.append(stats.pearsonr(y_true_control[:,class_index], y_score_control[:,class_index])[0])
            pearson_control = np.array(vals)
            print(pearson_control)
            
            print('Control Spearman rho')
            vals = []
            for class_index in range(y_true_control.shape[-1]):
                vals.append(stats.spearmanr(y_true_control[:,class_index], y_score_control[:,class_index])[0])
            spearman_control = np.array(vals)
            print(spearman_control)
            
            # Store metrics for plotting
            metrics_data['control'] = {
                'pearson_r': pearson_control,
                'spearman_r': spearman_control
            }
        else:
            print("Warning: Could not compute control metrics - y_true or y_score is None")
            pearson_control = np.array([])
            spearman_control = np.array([])
    
    print("=== All training completed ===")
    
    # Create comparison plots if we have metrics data
    if len(metrics_data) >= 2:  # Need at least 2 models to compare
        print("\nCreating performance comparison plots...")
        plot_metrics_comparison(metrics_data, plots_dir, expt_name)
    else:
        print("\nNot enough models trained to create comparison plots.")
        print("Need at least 2 models to compare performance.")
    
    # Summary of saved models
    print("\n" + "="*50)
    print("TWO-STAGE TRAINING SUMMARY")
    print("="*50)
    
    saved_models = []
    
    # Check Stage 1: Augmented model
    if os.path.exists(best_model_path):
        saved_models.append(f"✓ {os.path.basename(best_model_path)} - Stage 1: Model trained with EvoAug2 augmentations")
        saved_models.append("   → Learned robust representations from expanded genetic variation")
    else:
        saved_models.append(f"✗ Stage 1: NOT SAVED")
    
    # Check Stage 2: Fine-tuned model
    if os.path.exists(best_finetune_path):
        saved_models.append(f"✓ {os.path.basename(best_finetune_path)} - Stage 2: Model fine-tuned on original data")
        saved_models.append("   → Removed augmentation bias, refined features towards observed biology")
    else:
        saved_models.append(f"✗ Stage 2: NOT SAVED")
    
    # Check Control model
    if os.path.exists(best_control_path):
        saved_models.append(f"✓ {os.path.basename(best_control_path)} - Control: Model trained on original data only")
        saved_models.append("   → Baseline comparison for evaluating two-stage approach")
    else:
        saved_models.append(f"✗ Control: NOT SAVED")
    
    print("\nTraining Stages Completed:")
    for model in saved_models:
        print(f"  {model}")
    
    print(f"\nAll models saved to directory: {os.path.abspath(output_dir)}")
    print(f"All plots saved to directory: {os.path.abspath(plots_dir)}")


if __name__ == "__main__":
    main() 