# training_utils.py
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from enformer_pytorch import Enformer
from regex import D, F
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import PreTrainedModel

from deepseq.earlystopping import EarlyStopping
from deepseq.loss_calculation import TrainLossTracker, ValidationLossCalculator

DISTRIBUTED = int(os.environ.get("SM_NUM_GPUS", torch.cuda.device_count())) > 1


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
    train_loss_tracker: TrainLossTracker,
    val_loss_calculator: ValidationLossCalculator,
    early_stopping: EarlyStopping,
) -> float:
    model.train()

    for batch_idx, batch in enumerate(train_loader):
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

        if DISTRIBUTED:
            loss_tensor = torch.tensor([loss.item()], device=params.device)
            torch.distributed.all_reduce(loss_tensor)
            loss_tensor /= torch.distributed.get_world_size()
            loss_val = loss_tensor.item()
            rank = torch.distributed.get_rank()
        else:
            loss_val = loss.item()
            rank = 0

        if rank == 0:
            # Log the training loss to TensorBoard
            train_loss = train_loss_tracker(loss_val)

            # Calculate the validation loss and log it to TensorBoard
            val_loss, val_acc = val_loss_calculator(model)

            if train_loss and val_loss is not None:
                print(
                    f"Batch: {batch_idx}/{len(train_loader)} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {params.optimizer.param_groups[0]['lr']:.4f}"
                )

            if early_stopping(val_loss, model):
                print("Early stopping!")
                if DISTRIBUTED:
                    torch.distributed.destroy_process_group()
                return False

    return True
