from torchmetrics import MeanMetric, MinMetric, Accuracy, Precision, Recall, F1Score, ConfusionMatrix
import torch.nn as nn
import lightning.pytorch as pl
import torch

class SimpleMLP(pl.LightningModule):
    def __init__(self, model, optimizer, scheduler):
        super().__init__()
        input_dim = model.input_dim
        output_dim = model.output_dim
        hidden_channels = model.hidden_channels
        self.model = nn.ModuleList()
        for i in range(len(hidden_channels)):
            self.model.append(nn.Linear(input_dim, hidden_channels[i]))
            self.model.append(nn.ReLU())
            input_dim = hidden_channels[i]
        self.model.append(nn.Linear(input_dim, output_dim))
        
        self.optimizer_partial = optimizer
        self.scheduler_partial = scheduler
        
        self.criterion = nn.BCEWithLogitsLoss()
        
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.val_loss_best = MinMetric()
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.test_acc = Accuracy(task="binary")
        
        # Additional metrics for test evaluation
        self.test_precision = Precision(task="binary")
        self.test_recall = Recall(task="binary")
        self.test_f1 = F1Score(task="binary")
        self.test_confmat = ConfusionMatrix(task="binary", num_classes=2)
        
        self.save_hyperparameters()
        
    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x
    
    
    def on_train_start(self):
        """Lightning hook that is called when training begins."""
        self.train_loss.reset()
        self.val_loss.reset()
        self.val_loss_best.reset()
        self.train_acc.reset()
        self.val_acc.reset()
        self.test_acc.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_f1.reset()
        self.test_confmat.reset()
        
    def training_step(self, batch, batch_idx):
        x, y = batch["inputs"], batch["label"]
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.train_loss.update(loss)
        
        y_hat_sigmoid = torch.sigmoid(y_hat)
        self.train_acc.update(y_hat_sigmoid, y)
        self.log("train_loss", self.train_loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch["inputs"], batch["label"]
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.val_loss.update(loss)
        
        y_hat_sigmoid = torch.sigmoid(y_hat)
        self.val_acc.update(y_hat_sigmoid, y)
        self.log("val_loss", self.val_loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True)
        return loss
    
    def on_validation_epoch_end(self):
        self.val_loss_best.update(self.val_loss.compute())
        self.log("val_loss_best", self.val_loss_best, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        x, y = batch["inputs"], batch["label"]
        x[:, 4] = - 3 / 8
        x[:, 5] = 0.0
        x[:, 6] = 0.0
        x[:, 7] = 0.0
        
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # Apply sigmoid for metric calculations
        y_hat_sigmoid = torch.sigmoid(y_hat)
        
        # Update all metrics
        self.test_acc.update(y_hat_sigmoid, y)
        self.test_precision.update(y_hat_sigmoid, y)
        self.test_recall.update(y_hat_sigmoid, y)
        self.test_f1.update(y_hat_sigmoid, y)
        self.test_confmat.update(y_hat_sigmoid, y)
        
        # Log all metrics
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log("test_acc", self.test_acc, on_epoch=True, prog_bar=True)
        self.log("test_precision", self.test_precision, on_epoch=True, prog_bar=True)
        self.log("test_recall", self.test_recall, on_epoch=True, prog_bar=True)
        self.log("test_f1", self.test_f1, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_test_epoch_end(self):
        """Calculate and log TPR, FPR, TNR, FNR at the end of test epoch."""
        # Get confusion matrix values
        confmat = self.test_confmat.compute()
        tn, fp, fn, tp = confmat.flatten()
        
        # Calculate metrics
        tpr = tp / (tp + fn) if (tp + fn) > 0 else torch.tensor(0.0)  # same as recall
        fpr = fp / (fp + tn) if (fp + tn) > 0 else torch.tensor(0.0)
        tnr = tn / (tn + fp) if (tn + fp) > 0 else torch.tensor(0.0)  # specificity
        fnr = fn / (fn + tp) if (fn + tp) > 0 else torch.tensor(0.0)
        
        # Log metrics
        self.log("test_tpr", tpr, prog_bar=True)
        self.log("test_fpr", fpr, prog_bar=True)
        self.log("test_tnr", tnr, prog_bar=True)
        self.log("test_fnr", fnr, prog_bar=True)
        
        # Also log the confusion matrix components
        self.log("test_tp", tp, prog_bar=True)
        self.log("test_fp", fp, prog_bar=True)
        self.log("test_tn", tn, prog_bar=True)
        self.log("test_fn", fn, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        x = batch
        pred = self(x)
        return pred
    
    def configure_optimizers(self):
        self.optimizer = self.optimizer_partial(params=self.parameters())
        self.scheduler = self.scheduler_partial(optimizer=self.optimizer)
        
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "train_loss",
            },
        }