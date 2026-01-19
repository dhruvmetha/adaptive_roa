from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Dict, Any
from torchvision import transforms
from tqdm import tqdm
import os
import torch
import lightning.pytorch as pl
import numpy as np
import random
import omegaconf

class ReachabilityDataset(Dataset):
    def __init__(self, data: List[str], split="train"):
        """
        Args:
            data_dir: path to data directory
            transform: pytorch transforms for data augmentation
            split: one of 'train', 'val', or 'test'
        """
        self.split = split
        
        # split the data into train, val, test 
        if self.split == "train":
            samples = data[:int(0.8*len(data))]
        elif self.split == "val":
            samples = data[int(0.8*len(data)):int(0.9*len(data))]
        elif self.split == "test":
            samples = data[int(0.9*len(data)):]
    
        self.min_inputs = np.array([-4 * np.pi, -4 * np.pi, -20, -20] * 2)
        self.max_inputs = np.array([4 * np.pi, 4 * np.pi, 20, 20] * 2)
            
        self.samples = samples
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        sample = [float(x) for x in sample.strip().split(" ") if x != ""]
        inputs = torch.from_numpy((np.array(sample[:8]) - self.min_inputs) / (self.max_inputs - self.min_inputs)).float()
        label = torch.tensor([sample[-1]]).float()
        
        return {
            "inputs": inputs,
            "label": label
        }

class ReachabilityDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_file: str,
        test_data_file: str,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()
        
        self.data_file = data_file
        self.test_data_file = test_data_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        self.data = []
        
    def prepare_data(self):
        """
        preprocess data if needed
        """
        pass
   
        
    def setup(self, stage: Optional[str] = None):
        """
        Load data. Set variables: self.train_dataset, self.val_dataset, self.test_dataset
        """
        
        train_data = []
        with open(os.path.join(self.data_file, "train.txt"), 'r') as f:
            train_data = f.readlines()
        random.shuffle(train_data)
        
        val_data = []
        with open(os.path.join(self.data_file, "valid.txt"), 'r') as f:
            val_data = f.readlines()
        random.shuffle(val_data)
        
        test_data = []
        with open(self.test_data_file, 'r') as f:
            test_data = f.readlines()
        # random.shuffle(test_data)
        
        if stage == "fit" or stage is None:
            self.train_dataset = ReachabilityDataset(
                train_data, split="train"
            )
            
            self.val_dataset = ReachabilityDataset(
                val_data, split="val"
            )
            
        if stage == "test" or stage is None:
            self.test_dataset = ReachabilityDataset(
                test_data, split="test"
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
