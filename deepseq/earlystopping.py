from collections import deque
from typing import Optional, Protocol

import torch
from torch.cuda.amp import GradScaler
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader


def check_loss(model, dataloader, criterion, device, mode="train") -> float:
    if mode == "validate":
        model.eval()

    total_loss = 0.0

    with torch.set_grad_enabled(mode == "train"):
        for batch in dataloader:
            inputs, targets = batch[0].to(device), batch[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

    average_loss = total_loss / len(dataloader)

    if mode == "validate":
        model.train()

    return average_loss


class ModelParamsProtocol(Protocol):
    optimizer: Optimizer
    criterion: Module
    device: torch.device
    scaler: Optional[GradScaler]
    scheduler: Optional[_LRScheduler]


class EarlyStopping:
    def __init__(
        self,
        model,
        val_dataloader: DataLoader,
        model_params: ModelParamsProtocol,
        patience: int,
        verbose: bool = False,
        delta: float = 0,
        save_path: Optional[str] = None,
        check_frequency: int = 1,
    ):
        self.patience = patience
        self.verbose = verbose
        self.patience_counter = 0
        self.best_loss = float("inf")
        self.delta = delta
        self.save_path = save_path
        self.model_params = model_params
        self.check_frequency = check_frequency
        self.frequency_counter = 0

        self.model = model
        self.val_dataloader = val_dataloader

    def __call__(self, model):
        self.frequency_counter += 1

        if self.frequency_counter < self.check_frequency:
            return False

        self.frequency_counter = 0

        # Compute validation loss
        val_loss = check_loss(
            self.model,
            self.val_dataloader,
            self.model_params.criterion,
            self.model_params.device,
            mode="validate",
        )

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
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
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

        self.best_loss = val_loss  # Update the best_loss to current validation loss
