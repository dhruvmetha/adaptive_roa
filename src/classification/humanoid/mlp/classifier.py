import torch
import torch.nn as nn
import lightning.pytorch as pl
from torchmetrics import Accuracy, AUROC, Precision, Recall, F1Score, Specificity
from typing import Optional, Dict, Any

from src.systems.humanoid import HumanoidSystem
from src.model.baseline_classifier import MLPClassifier

class HumanoidBaselineClassifier(pl.LightningModule):
    """
    Lightning Module for Humanoid ROA Classification Baseline.
    Uses HumanoidSystem for normalization and an MLP for classification.
    """
    def __init__(self, 
                 system: HumanoidSystem,
                 model_config: Dict[str, Any],
                 learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=['system'])
        self.system = system
        self.lr = learning_rate
        
        # Create model
        self.model = MLPClassifier(
            input_dim=67, # Humanoid state dim
            hidden_dims=model_config.get('hidden_dims', [128, 128]),
            dropout=model_config.get('dropout', 0.1),
            activation=model_config.get('activation', 'relu')
        )
        
        # Loss function (Binary Cross Entropy with Logits)
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Metrics
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.test_acc = Accuracy(task="binary")
        
        self.val_auroc = AUROC(task="binary")
        self.test_auroc = AUROC(task="binary")
        
        self.test_precision = Precision(task="binary")
        self.test_recall = Recall(task="binary")
        self.test_specificity = Specificity(task="binary")
        self.test_f1 = F1Score(task="binary")

    def forward(self, x):
        # Normalize input using system definition
        # Note: Training data is raw, so we normalize before feeding to MLP
        norm_x = self.system.normalize_state(x)
        return self.model(norm_x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.sigmoid(logits)
        self.train_acc(preds, y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.sigmoid(logits)
        self.val_acc(preds, y)
        self.val_auroc(preds, y)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, prog_bar=True)
        self.log('val_auroc', self.val_auroc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.sigmoid(logits)
        self.test_acc(preds, y)
        self.test_auroc(preds, y)
        self.test_precision(preds, y)
        self.test_recall(preds, y)
        self.test_specificity(preds, y)
        self.test_f1(preds, y)
        
        self.log('test_loss', loss)
        self.log('test_acc', self.test_acc)
        self.log('test_auroc', self.test_auroc)
        self.log('test_precision', self.test_precision)
        self.log('test_recall', self.test_recall)
        self.log('test_specificity', self.test_specificity)
        self.log('test_f1', self.test_f1)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
