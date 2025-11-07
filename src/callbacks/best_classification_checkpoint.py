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
            save_top_k: 1
            mode: max
    """

    def __init__(
        self,
        dirpath: Optional[str] = None,
        filename: Optional[str] = None,
        save_top_k: int = 1,
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
        # We don't monitor a specific metric in __init__ because we'll compute it dynamically
        super().__init__(
            dirpath=dirpath,
            filename=filename,
            monitor=None,  # We'll handle monitoring manually
            save_top_k=save_top_k,
            mode='max',
            **kwargs
        )

        self.best_score = float('-inf')

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Called at the end of validation epoch.
        Compute classification metric sum and save checkpoint if it's the best.
        """
        # Get the current metrics from the model
        if hasattr(pl_module, 'test_precision') and hasattr(pl_module, 'test_recall') and hasattr(pl_module, 'test_specificity'):
            try:
                # Check if metrics have been updated by checking if they have samples
                # We check the internal state of torchmetrics - different attributes for different versions
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

                    # Log the mean score
                    pl_module.log('test_classification_mean', current_score, on_step=False, on_epoch=True, prog_bar=True)

                    # Check if this is the best score
                    if current_score > self.best_score:
                        self.best_score = current_score

                        # Save checkpoint
                        filepath = self._get_metric_interpolated_filepath_name(
                            {
                                'epoch': trainer.current_epoch,
                                'test_precision': precision,
                                'test_recall': recall,
                                'test_specificity': specificity,
                                'test_classification_mean': current_score,
                            },
                            trainer,
                            {}
                        )

                        self._save_checkpoint(trainer, filepath)

                        print(f"\n💾 New best classification model saved!")
                        print(f"   Mean Score: {current_score:.4f} (Precision: {precision:.4f}, Recall: {recall:.4f}, Specificity: {specificity:.4f})")
                        print(f"   Checkpoint: {filepath}")

            except Exception as e:
                # Metrics might not be available yet (e.g., first epoch before MAE computation)
                # Silently skip - this is expected in early epochs
                pass
