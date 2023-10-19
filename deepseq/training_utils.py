import math
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from enformer_pytorch import Enformer
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel

from deepseq.earlystopping import EarlyStopping


def count_directories(path: str) -> int:
    # Check if path exists and is a directory
    assert os.path.exists(path), "The specified path does not exist."
    assert os.path.isdir(path), "The specified path is not a directory."

    # Count only directories within the specified path
    directory_count = sum(
        os.path.isdir(os.path.join(path, i)) for i in os.listdir(path)
    )
    return directory_count


def transfer_enformer_weights_to_(
    model: PreTrainedModel, transformer_only: bool = False
) -> PreTrainedModel:
    # Load pretrained weights
    enformer = Enformer.from_pretrained("EleutherAI/enformer-official-rough")

    if transformer_only:  # pretty much if num_downsamples is not 7 use this
        # Create a new state dict excluding keys related to enformer.stem and enformer._trunk
        excluded_prefixes = ["stem.", "_trunk.", "conv_tower"]
        state_dict_to_load = {
            key: value
            for key, value in enformer.state_dict().items()
            if not any(key.startswith(prefix) for prefix in excluded_prefixes)
        }
        model.load_state_dict(state_dict_to_load, strict=False)
    else:
        # Create a new state dict excluding keys related to enformer.stem and enformer._trunk
        excluded_prefixes = ["stem.", "_trunk."]
        state_dict_to_load = {
            key: value
            for key, value in enformer.state_dict().items()
            if not any(key.startswith(prefix) for prefix in excluded_prefixes)
        }
        model.load_state_dict(state_dict_to_load, strict=False)

    return model


class WarmupCosineSchedule(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super(WarmupCosineSchedule, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            return [base_lr * step / self.warmup_steps for base_lr in self.base_lrs]
        return [
            base_lr
            * (
                0.5
                * (
                    1.0
                    + math.cos(
                        math.pi
                        * (step - self.warmup_steps)
                        / (self.total_steps - self.warmup_steps)
                    )
                )
            )
            for base_lr in self.base_lrs
        ]


@dataclass
class TrainingParams:
    optimizer: torch.optim.Optimizer
    criterion: nn.Module
    device: torch.device
    scaler: Optional[GradScaler]
    scheduler: Optional[_LRScheduler] = None


def train_one_epoch(
    model,
    params: TrainingParams,
    train_loader: DataLoader,
    early_stopping: EarlyStopping,
) -> float:
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(train_loader, desc="Training")

    for batch_idx, batch in enumerate(progress_bar):
        inputs, targets = batch[0].to(params.device), batch[1].to(params.device)

        params.optimizer.zero_grad()

        # If scaler is provided, use mixed precision
        with torch.cuda.amp.autocast(enabled=params.scaler is not None):
            outputs = model(inputs)
            loss = params.criterion(outputs, targets)

        # If scaler is provided, scale the loss
        if params.scaler:
            params.scaler.scale(loss).backward()
            params.scaler.step(params.optimizer)
            params.scaler.update()
        else:
            loss.backward()
            params.optimizer.step()

        # Step the scheduler if provided
        if params.scheduler:
            params.scheduler.step()

        total_loss += loss.item()

        # Update tqdm
        progress_bar.refresh()

        if early_stopping(model):
            print("Early stopping!")
            break

    average_loss = total_loss / len(train_loader)
    return average_loss
