import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split, Subset, WeightedRandomSampler
import lightning.pytorch as pl
from typing import Optional, Tuple, Union, List
from omegaconf import ListConfig

class HumanoidROADataset(Dataset):
    """
    Dataset for Humanoid ROA labeled data.
    """
    def __init__(self, data_file: str):
        """
        Args:
            data_file: Path to roa_labels.txt
        """
        self.data_file = data_file
        
        # Load data
        # Format is assumed to be comma-separated: 67D state + 1 label
        raw_data = np.loadtxt(data_file, delimiter=',')
        
        # Split into states and labels
        # Last column is label
        self.states = torch.from_numpy(raw_data[:, :-1]).float()
        self.labels = torch.from_numpy(raw_data[:, -1]).float()
        
        print(f"Loaded {len(self.states)} samples from {data_file}")
        print(f"State shape: {self.states.shape}")
        print(f"Label balance: {(self.labels == 1).sum()} success / {(self.labels == 0).sum()} failure")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.labels[idx].unsqueeze(0) # (1,) label

class HumanoidROADataModule(pl.LightningDataModule):
    """
    DataModule for splitting ROA data into train/val/test.
    Supports both percentage-based random splitting and fixed index slicing.
    Implements WeightedRandomSampling for class balance in training.
    """
    def __init__(self, 
                 data_file: str,
                 batch_size: int = 64,
                 num_workers: int = 4,
                 train_split: Union[float, List[int]] = 0.8,
                 val_split: Union[float, List[int]] = 0.1,
                 test_split: Union[float, List[int]] = 0.1,
                 seed: int = 42,
                 use_weighted_sampler: bool = True):
        """
        Args:
            data_file: Path to data file
            batch_size: Batch size
            num_workers: Number of workers
            train_split: If float, fraction of data. If list [start, end], specific indices.
            val_split: If float, fraction of data. If list [start, end], specific indices.
            test_split: If float, fraction of data. If list [start, end], specific indices.
            seed: Random seed (only used for percentage splits)
            use_weighted_sampler: If True, uses WeightedRandomSampler for training
        """
        super().__init__()
        self.data_file = data_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.splits = {'train': train_split, 'val': val_split, 'test': test_split}
        self.seed = seed
        self.use_weighted_sampler = use_weighted_sampler
        self.train_weights = None
        
    def setup(self, stage: Optional[str] = None):
        # Load full dataset
        full_dataset = HumanoidROADataset(self.data_file)
        total_len = len(full_dataset)
        
        # Check if we are using manual slicing (lists) or random fractions (floats)
        # omegaconf.ListConfig is what Hydra passes for lists in yaml
        is_list = lambda x: isinstance(x, (list, tuple, ListConfig))
                             
        if is_list(self.splits['train']):
             # Manual slicing mode
             train_indices = list(range(self.splits['train'][0], self.splits['train'][1]))
             
             if is_list(self.splits['val']):
                 val_indices = list(range(self.splits['val'][0], self.splits['val'][1]))
             else:
                 raise ValueError("If train_split is a list [start, end], val_split and test_split must also be lists.")

             if is_list(self.splits['test']):
                 test_indices = list(range(self.splits['test'][0], self.splits['test'][1]))
             else:
                 raise ValueError("If train_split is a list [start, end], val_split and test_split must also be lists.")
                 
             self.train_ds = Subset(full_dataset, train_indices)
             self.val_ds = Subset(full_dataset, val_indices)
             self.test_ds = Subset(full_dataset, test_indices)
             
             print(f"Using manual slicing:")
             print(f"  Train: indices {self.splits['train']} ({len(self.train_ds)} samples)")
             print(f"  Val:   indices {self.splits['val']} ({len(self.val_ds)} samples)")
             print(f"  Test:  indices {self.splits['test']} ({len(self.test_ds)} samples)")
             
        else:
            # Random percentage split mode
            train_len = int(total_len * self.splits['train'])
            val_len = int(total_len * self.splits['val'])
            test_len = total_len - train_len - val_len
            
            generator = torch.Generator().manual_seed(self.seed)
            self.train_ds, self.val_ds, self.test_ds = random_split(
                full_dataset, [train_len, val_len, test_len], generator=generator
            )
            
            print(f"Using random split:")
            print(f"  Train: {len(self.train_ds)} samples")
            print(f"  Val:   {len(self.val_ds)} samples")
            print(f"  Test:  {len(self.test_ds)} samples")

        # Calculate weights for weighted sampling
        if self.use_weighted_sampler:
            print("⚖️ Calculating sample weights for balanced training...")
            
            # Get labels for training set
            # Note: Subset or random_split dataset doesn't expose labels directly easily, 
            # we need to access via indices if it's a Subset
            if isinstance(self.train_ds, Subset):
                train_indices = self.train_ds.indices
            else:
                train_indices = self.train_ds.indices # random_split also returns a Subset-like object with indices
                
            train_labels = full_dataset.labels[train_indices]
            
            # Count classes
            num_success = (train_labels == 1).sum().item()
            num_failure = (train_labels == 0).sum().item()
            total = len(train_labels)
            
            print(f"   Training set balance: {num_success} Success, {num_failure} Failure")
            
            if num_success > 0 and num_failure > 0:
                weight_success = 1.0 / num_success
                weight_failure = 1.0 / num_failure
                
                # Assign weight to each sample
                self.train_weights = torch.zeros(total)
                self.train_weights[train_labels == 1] = weight_success
                self.train_weights[train_labels == 0] = weight_failure
                
                print(f"   Weights: Success={weight_success:.6f}, Failure={weight_failure:.6f}")
            else:
                print("   ⚠️ Warning: One class has 0 samples in training set. Disabling weighted sampling.")
                self.use_weighted_sampler = False

    def train_dataloader(self):
        if self.use_weighted_sampler and self.train_weights is not None:
            sampler = WeightedRandomSampler(
                weights=self.train_weights,
                num_samples=len(self.train_weights),
                replacement=True
            )
            return DataLoader(self.train_ds, batch_size=self.batch_size, sampler=sampler, num_workers=self.num_workers)
        else:
            return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
