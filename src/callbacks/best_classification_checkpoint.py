"""
Custom ModelCheckpoint callback that saves the best model based on classification metrics.

Monitors the mean of (precision + recall + specificity) / 3 and saves the model with the highest score.
"""
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from typing import Optional


class BestClassificationCheckpoint(ModelCheckpoint):
    """
    ModelCheckpoint that saves the best model based on classification metrics mean.

    This callback monitors (precision + recall + specificity) / 3 and saves the model
    with the highest mean score.

    Usage in config:
        callbacks:
          - _target_: src.callbacks.best_classification_checkpoint.BestClassificationCheckpoint
            dirpath: "${hydra:runtime.output_dir}/version_0/checkpoints"
            filename: "best_classification-{epoch:02d}-{test_classification_mean:.4f}"
            save_top_k: 3
            mode: max
    """

    def __init__(
        self,
        dirpath: Optional[str] = None,
        filename: Optional[str] = None,
        save_top_k: int = 3,
        **kwargs
    ):
        """
        Initialize the best classification checkpoint callback.

        Args:
            dirpath: Directory to save checkpoints
            filename: Checkpoint filename format
            save_top_k: Number of best models to save
            **kwargs: Additional arguments for ModelCheckpoint
        """
        # Monitor the test_classification_mean metric
        super().__init__(
            dirpath=dirpath,
            filename=filename,
            monitor='test_classification_mean',
            save_top_k=save_top_k,
            mode='max',
            **kwargs
        )

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Called at the end of validation epoch.
        Compute classification metric mean and let parent class handle checkpoint saving.
        """
        # Get the current metrics from the model
        if hasattr(pl_module, 'test_precision') and hasattr(pl_module, 'test_recall') and hasattr(pl_module, 'test_specificity'):
            try:
                # Check if metrics have been updated by checking if they have samples
                has_samples = False
                if hasattr(pl_module.test_precision, 'tp'):
                    # Check if any true positives, false positives, etc. have been recorded
                    has_samples = (pl_module.test_precision.tp + pl_module.test_precision.fp +
                                  pl_module.test_precision.fn + pl_module.test_precision.tn) > 0
                elif hasattr(pl_module.test_precision, '_update_count'):
                    has_samples = pl_module.test_precision._update_count > 0

                if has_samples:
                    precision = pl_module.test_precision.compute().item()
                    recall = pl_module.test_recall.compute().item()
                    specificity = pl_module.test_specificity.compute().item()

                    # Compute mean score
                    current_score = (precision + recall + specificity) / 3.0

                    # Log the mean score (this is what we monitor)
                    pl_module.log('test_classification_mean', current_score, on_step=False, on_epoch=True, prog_bar=True)

                    # Let the parent class handle saving based on the monitored metric
                    super().on_validation_epoch_end(trainer, pl_module)

            except Exception as e:
                # Metrics might not be available yet (e.g., first epoch before MAE computation)
                # Silently skip - this is expected in early epochs
                pass
            