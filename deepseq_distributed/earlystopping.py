from collections import deque
from typing import Optional, Protocol

import torch
from torch.cuda.amp import GradScaler
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader


class EarlyStopping:
    def __init__(
        self,
        patience: int,
        verbose: bool = False,
        delta: float = 0,
        save_path: Optional[str] = None,
    ):
        self.patience = patience
        self.verbose = verbose
        self.patience_counter = 0
        self.best_loss = float("inf")
        self.delta = delta
        self.save_path = save_path

    def __call__(self, val_loss, model):
        # If the validation loss is not calculated, do nothing
        if val_loss is None:
            return False

        # Check if the current loss has improved sufficiently
        if val_loss > self.best_loss - self.delta:
            self.patience_counter += 1
            if self.verbose:
                print(
                    f"EarlyStopping counter: {self.patience_counter} out of {self.patience}"
                )

            # Signal to stop training after hitting the patience limit
            if self.patience_counter >= self.patience:
                return True
        else:
            # Improvement in the loss, reset the counter and save the current model
            self.save_checkpoint(val_loss, model)
            self.best_loss = val_loss
            self.patience_counter = 0

        return False  # Continue training

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.save_path:
            torch.save(model.state_dict(), self.save_path)
        if self.verbose:
            print(
                f"Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).  Saving model ..."
            )

        self.best_loss = val_loss
