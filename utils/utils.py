# From evoaug_analysis.git by p-koo

import os, pathlib, h5py
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error
from scipy import stats
import torch
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset


#------------------------------------------------------------------------
# useful functions
#------------------------------------------------------------------------


def make_directory(directory):
    """make directory"""
    if not os.path.isdir(directory):
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        print("Making directory: " + directory)


def evaluate_model(y_test, pred, task, verbose=True):
    if task == 'regression': #isinstance(pl_model.criterion, torch.nn.modules.loss.MSELoss):
        mse = calculate_mse(y_test, pred)
        pearsonr = calculate_pearsonr(y_test, pred)
        spearmanr = calculate_spearmanr(y_test, pred)
        if verbose:
            print("Test MSE       : %.4f +/- %.4f"%(np.nanmean(mse), np.nanstd(mse)))
            print("Test Pearson r : %.4f +/- %.4f"%(np.nanmean(pearsonr), np.nanstd(pearsonr)))
            print("Test Spearman r: %.4f +/- %.4f"%(np.nanmean(spearmanr), np.nanstd(spearmanr)))
        return mse, pearsonr, spearmanr

    else: 
        auroc = calculate_auroc(y_test, pred) 
        aupr = calculate_aupr(y_test, pred) 
        if verbose:
            print("Test AUROC: %.4f +/- %.4f"%(np.nanmean(auroc), np.nanstd(auroc)))
            print("Test AUPR : %.4f +/- %.4f"%(np.nanmean(aupr), np.nanstd(aupr)))
        return auroc, aupr


def calculate_auroc(y_true, y_score):
    vals = []
    for class_index in range(y_true.shape[-1]):
        vals.append( roc_auc_score(y_true[:,class_index], y_score[:,class_index]) )    
    return np.array(vals)

def calculate_aupr(y_true, y_score):
    vals = []
    for class_index in range(y_true.shape[-1]):
        vals.append( average_precision_score(y_true[:,class_index], y_score[:,class_index]) )    
    return np.array(vals)

def calculate_mse(y_true, y_score):
    vals = []
    for class_index in range(y_true.shape[-1]):
        vals.append( mean_squared_error(y_true[:,class_index], y_score[:,class_index]) )    
    return np.array(vals)

def calculate_pearsonr(y_true, y_score):
    vals = []
    for class_index in range(y_true.shape[-1]):
        vals.append( stats.pearsonr(y_true[:,class_index], y_score[:,class_index])[0] )    
    return np.array(vals)
    
def calculate_spearmanr(y_true, y_score):
    vals = []
    for class_index in range(y_true.shape[-1]):
        vals.append( stats.spearmanr(y_true[:,class_index], y_score[:,class_index])[0] )    
    return np.array(vals)


#------------------------------------------------------------------------
# useful pytorch functions
#------------------------------------------------------------------------


def configure_optimizer(model, lr=0.001, weight_decay=1e-6, decay_factor=0.1, patience=5, monitor='val_loss'):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=decay_factor, patience=patience),
            "monitor": monitor,
        },
    }


def get_predictions(model, x, batch_size=100, accelerator='gpu', devices=1):
    """Get predictions from a PyTorch model (not a Lightning module)."""
    # trainer = pl.Trainer(accelerator=accelerator, devices=devices, logger=None)
    # pred = trainer.predict(model, dataloaders=dataloader)
    # return np.concatenate(pred)
    model.eval()
    predictions = []
    
    # Convert x to tensor if it's not already
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    
    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() and accelerator == 'gpu' else 'cpu')
    model = model.to(device)
    
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            batch_x = x[i:i+batch_size].to(device)
            batch_pred = model(batch_x)
            predictions.append(batch_pred.cpu())
    
    return torch.cat(predictions, dim=0).numpy()



def get_fmaps(robust_model, x):
    """Get first layer feature maps -- must be named -- activation1"""
    fmaps = []
    def get_output(the_list):
        """get output of layer and put it into list the_list"""
        def hook(model, input, output):
            the_list.append(output.data);
        return hook

    robust_model = robust_model.eval().to(torch.device("cpu")) # move back to CPU
    handle = robust_model.model.activation1.register_forward_hook(get_output(fmaps))
    with torch.no_grad():
        robust_model.model(x);
    handle.remove()
    return fmaps[0].detach().cpu().numpy().transpose([0,2,1])



#------------------------------------------------------------------------
# Generic Dataloader for pytorch
#------------------------------------------------------------------------


class H5DataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size=128, stage=None, lower_case=False, transpose=False, downsample=None):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.x = 'X'
        self.y = 'Y'
        if lower_case:
            self.x = 'x'
            self.y = 'y'
        self.transpose = transpose
        self.downsample = downsample
        self.setup(stage)

    def setup(self, stage=None):
        # Assign train and val split(s) for use in DataLoaders
        if stage == "fit" or stage is None:
            with h5py.File(self.data_path, 'r') as dataset:
                x_train = np.array(dataset[self.x+"_train"]).astype(np.float32)
                y_train = np.array(dataset[self.y+"_train"]).astype(np.float32)
                x_valid = np.array(dataset[self.x+"_valid"]).astype(np.float32)
                if self.transpose:
                    x_train = np.transpose(x_train, (0,2,1))
                    x_valid = np.transpose(x_valid, (0,2,1))
                if self.downsample:
                    x_train = x_train[:self.downsample]
                    y_train = y_train[:self.downsample]
                self.x_train = torch.from_numpy(x_train)
                self.y_train = torch.from_numpy(y_train)
                self.x_valid = torch.from_numpy(x_valid)
                self.y_valid = torch.from_numpy(np.array(dataset[self.y+"_valid"]).astype(np.float32))
            _, self.A, self.L = self.x_train.shape # N = number of seqs, A = alphabet size (number of nucl.), L = length of seqs
            self.num_classes = self.y_train.shape[1]
            
        # Assign test split(s) for use in DataLoaders
        if stage == "test" or stage is None:
            with h5py.File(self.data_path, "r") as dataset:
                x_test = np.array(dataset[self.x+"_test"]).astype(np.float32)
                if self.transpose:
                    x_test = np.transpose(x_test, (0,2,1))
                self.x_test = torch.from_numpy(x_test)
                self.y_test = torch.from_numpy(np.array(dataset[self.y+"_test"]).astype(np.float32))
            _, self.A, self.L = self.x_train.shape
            self.num_classes = self.y_train.shape[1]
            
    def train_dataloader(self):
        train_dataset = TensorDataset(self.x_train, self.y_train) # tensors are index-matched
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True) # sets of (x, x', y) will be shuffled
    
    def val_dataloader(self):
        valid_dataset = TensorDataset(self.x_valid, self.y_valid) 
        return DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self):
        test_dataset = TensorDataset(self.x_test, self.y_test)
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False) 


class H5Dataset(Dataset):
    """
    Enhanced Dataset class for H5 data files with DataModule-like functionality.
    
    This class combines the functionality of a PyTorch Dataset with the convenience
    methods of a Lightning DataModule, making it easy to integrate with EvoAug2
    augmentations without nesting datamodules.
    
    Parameters
    ----------
    filepath : str
        Path to the H5 file
    batch_size : int, optional
        Batch size for dataloaders. Defaults to 128.
    lower_case : bool, optional
        Whether to use lowercase keys ('x', 'y') instead of uppercase ('X', 'Y').
        Defaults to False.
    transpose : bool, optional
        Whether to transpose the data dimensions. Defaults to False.
    downsample : int, optional
        Number of samples to use (for debugging). If None, uses all data.
        Defaults to None.
    """
    
    def __init__(self, filepath, batch_size=128, lower_case=False, transpose=False, downsample=None):
        self.filepath = filepath
        self.batch_size = batch_size
        self.lower_case = lower_case
        self.transpose = transpose
        self.downsample = downsample
        
        # Set key names based on lower_case parameter
        self.x_key = 'x' if lower_case else 'X'
        self.y_key = 'y' if lower_case else 'Y'
        
        # Initialize data attributes
        self.x_train = None
        self.y_train = None
        self.x_valid = None
        self.y_valid = None
        self.x_test = None
        self.y_test = None
        
        # Initialize shape attributes
        self.A = None  # alphabet size (number of nucleotides)
        self.L = None  # sequence length
        self.num_classes = None
        
        # Load all data splits
        self._load_all_data()
    
    def _load_all_data(self):
        """Load all data splits from the H5 file."""
        with h5py.File(self.filepath, 'r') as f:
            # Load training data
            x_train = np.array(f[f'{self.x_key}_train'][:], dtype=np.float32)
            y_train = np.array(f[f'{self.y_key}_train'][:], dtype=np.float32)
            
            # Load validation data
            x_valid = np.array(f[f'{self.x_key}_valid'][:], dtype=np.float32)
            y_valid = np.array(f[f'{self.y_key}_valid'][:], dtype=np.float32)
            
            # Load test data
            x_test = np.array(f[f'{self.x_key}_test'][:], dtype=np.float32)
            y_test = np.array(f[f'{self.y_key}_test'][:], dtype=np.float32)
            
            # Apply downsampling if specified
            if self.downsample:
                x_train = x_train[:self.downsample]
                y_train = y_train[:self.downsample]
            
            # Apply transpose if needed
            if self.transpose:
                x_train = np.transpose(x_train, (0, 2, 1))
                x_valid = np.transpose(x_valid, (0, 2, 1))
                x_test = np.transpose(x_test, (0, 2, 1))
            
            # Convert to tensors
            self.x_train = torch.from_numpy(x_train)
            self.y_train = torch.from_numpy(y_train)
            self.x_valid = torch.from_numpy(x_valid)
            self.y_valid = torch.from_numpy(y_valid)
            self.x_test = torch.from_numpy(x_test)
            self.y_test = torch.from_numpy(y_test)
            
            # Set shape attributes
            _, self.A, self.L = self.x_train.shape
            self.num_classes = self.y_train.shape[1]
    
    def setup(self, split='train'):
        """
        Set up the dataset for a specific split.
        
        This method is kept for backward compatibility but is no longer needed
        as all data is loaded in __init__.
        
        Parameters
        ----------
        split : str, optional
            Data split to use ('train', 'val', 'test'). Defaults to 'train'.
            
        Notes
        -----
        This method is deprecated. All data is now loaded automatically in __init__.
        """
        # For backward compatibility, set current split
        if split == 'train':
            self.x = self.x_train
            self.y = self.y_train
        elif split == 'val':
            self.x = self.x_valid
            self.y = self.y_valid
        elif split == 'test':
            self.x = self.x_test
            self.y = self.y_test
        else:
            raise ValueError(f"Unknown split: {split}")
    
    def get_train_dataset(self):
        """Get training dataset as TensorDataset.
        
        Returns
        -------
        torch.utils.data.TensorDataset
            Training dataset with (x, y) pairs.
        """
        return TensorDataset(self.x_train, self.y_train)
    
    def get_val_dataset(self):
        """Get validation dataset as TensorDataset.
        
        Returns
        -------
        torch.utils.data.TensorDataset
            Validation dataset with (x, y) pairs.
        """
        return TensorDataset(self.x_valid, self.y_valid)
    
    def get_test_dataset(self):
        """Get test dataset as TensorDataset.
        
        Returns
        -------
        torch.utils.data.TensorDataset
            Test dataset with (x, y) pairs.
        """
        return TensorDataset(self.x_test, self.y_test)
    
    def train_dataloader(self):
        """Get training dataloader.
        
        Returns
        -------
        torch.utils.data.DataLoader
            Training dataloader with shuffling enabled.
        """
        train_dataset = self.get_train_dataset()
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        """Get validation dataloader.
        
        Returns
        -------
        torch.utils.data.DataLoader
            Validation dataloader with shuffling disabled.
        """
        valid_dataset = self.get_val_dataset()
        return DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self):
        """Get test dataloader.
        
        Returns
        -------
        torch.utils.data.DataLoader
            Test dataloader with shuffling disabled.
        """
        test_dataset = self.get_test_dataset()
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
    
    def __len__(self):
        """Return the number of samples in the current split.
        
        Returns
        -------
        int
            Number of samples in the current split.
        """
        if hasattr(self, 'x'):
            return len(self.x)
        else:
            # Default to training set length
            return len(self.x_train)
    
    def __getitem__(self, idx):
        """Get a single sample from the current split.
        
        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.
            
        Returns
        -------
        tuple
            (x, y) pair where x is the sequence and y is the target.
        """
        if hasattr(self, 'x'):
            return self.x[idx], self.y[idx]
        else:
            # Default to training set
            return self.x_train[idx], self.y_train[idx]
    
    @property
    def train_size(self):
        """Number of training samples."""
        return len(self.x_train)
    
    @property
    def val_size(self):
        """Number of validation samples."""
        return len(self.x_valid)
    
    @property
    def test_size(self):
        """Number of test samples."""
        return len(self.x_test)
    
    @property
    def total_size(self):
        """Total number of samples across all splits."""
        return self.train_size + self.val_size + self.test_size