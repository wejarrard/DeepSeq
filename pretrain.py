import argparse
import os
import warnings
from dataclasses import dataclass
from multiprocessing import cpu_count

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup

from deepseq.data import MaskedGenomeIntervalDataset
from deepseq.deepseq import DeepSeq
from deepseq.earlystopping import EarlyStopping
from deepseq.finetune import HeadAdapterWrapper
from deepseq.loss import FocalLoss
from deepseq.loss_calculation import TrainLossTracker, ValidationLossCalculator
from deepseq.training_utils import (
    TrainingParams,
    count_directories,
    train_one_epoch,
    transfer_enformer_weights_to_,
)

# hide user warning
warnings.filterwarnings("ignore", category=UserWarning)


############ HYPERPARAMETERS ############
@dataclass
class HyperParams:
    num_epochs: int = 2
    batch_size: int = 32 * torch.cuda.device_count() if torch.cuda.is_available() else 4
    learning_rate: float = 1e-4
    early_stopping_patience: int = 10
    validation_check_frequency: int = 5_000 if torch.cuda.is_available() else 4
    focal_loss_alpha: float = 1
    focal_loss_gamma: float = 2


def main(output_dir: str, data_dir: str, hyperparams: HyperParams) -> None:
    ############ DEVICE ############

    gpu_ok = torch.cuda.is_available() and torch.cuda.get_device_capability() in (
        (7, 0),
        (8, 0),
        (9, 0),
    )

    if not gpu_ok:
        warnings.warn(
            "GPU is not NVIDIA V100, A100, or H100. Speedup numbers may be lower than expected."
        )
    # Initialize model and data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ############ MODEL ############

    num_cell_lines = count_directories(os.path.join(data_dir, "cell_lines/"))

    deepseq = DeepSeq.from_hparams(
        dim=1536,
        depth=11,
        heads=8,
        target_length=-1,
        return_augs=True,
        rc_aug=True,
        num_downsamples=5,
    ).to(device)

    deepseq = transfer_enformer_weights_to_(
        deepseq, transformer_only=True
    )  # NOTE: We are not going to freeze any weights, can try that later (maybe only the transformer layer)

    head_out = nn.Sequential(
        Rearrange("b t c -> b c t"),
        nn.AvgPool1d(8),
        Rearrange("b c t -> b (c t)"),  # Flatten channel and time dimensions
        nn.Linear(in_features=128, out_features=num_cell_lines),
        nn.Sigmoid(),
    )

    model = HeadAdapterWrapper(
        enformer=deepseq,
        num_tracks=num_cell_lines,
        post_transformer_embed=True,
        output_activation=head_out,
    )

    if torch.cuda.device_count() > 1:
        pass

    model = model.to(device)

    model = torch.compile(model) if gpu_ok else model

    ############ DATA ############

    dataset = MaskedGenomeIntervalDataset(
        bed_file=os.path.join(data_dir, "consolidated.bed"),
        fasta_file=os.path.join(data_dir, "genome.fa"),
        cell_lines_dir=os.path.join(data_dir, "cell_lines/"),
        return_augs=False,
        rc_aug=False,
        shift_augs=(-10, 10),
        mask_prob=0.15,
        context_length=16_384,
    )

    total_size = len(dataset)
    valid_size = 20000 if torch.cuda.is_available() else 8
    train_size = total_size - valid_size

    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    assert (
        train_size > 0
    ), f"The dataset only contains {total_size} samples, but {valid_size} samples are required for the validation set."

    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    num_workers = cpu_count() if torch.cuda.is_available() else 0
    print(f"Using {num_workers} workers")

    train_loader = DataLoader(
        train_dataset,
        batch_size=hyperparams.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=hyperparams.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    ############ TRAINING PARAMS ############

    criterion = FocalLoss(
        alpha=hyperparams.focal_loss_alpha, gamma=hyperparams.focal_loss_gamma
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hyperparams.learning_rate,
        weight_decay=0.1,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scaler = GradScaler()

    total_steps = len(train_loader) * hyperparams.num_epochs
    warmup_steps = 0.1 * total_steps if torch.cuda.is_available() else 10

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    training_params = TrainingParams(
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scaler=scaler,
        scheduler=scheduler,
    )

    early_stopping = EarlyStopping(
        patience=hyperparams.early_stopping_patience,
        verbose=True,
        save_path=os.path.join(output_dir, "best_model.pth"),
    )

    ############ TENSORBOARD ############

    writer = SummaryWriter(
        log_dir="/opt/ml/output/tensorboard" if torch.cuda.is_available() else "output"
    )

    train_loss_tracker = TrainLossTracker(
        criterion=criterion,
        device=device,
        writer=writer,
        check_frequency=hyperparams.validation_check_frequency,
    )

    val_loss_calculator = ValidationLossCalculator(
        val_dataloader=valid_loader,
        criterion=criterion,
        device=device,
        writer=writer,
        check_frequency=hyperparams.validation_check_frequency,
    )

    ############ TRAINING ############

    for epoch in range(hyperparams.num_epochs):
        continue_training = train_one_epoch(
            model=model,
            params=training_params,
            train_loader=train_loader,
            early_stopping=early_stopping,
            train_loss_tracker=train_loss_tracker,
            val_loss_calculator=val_loss_calculator,
        )
        print("-" * 80)
        if not continue_training:
            print("Early stopping criterion met. Stopping training.")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DeepSeq model on SageMaker.")
    parser.add_argument(
        "--output-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR")
    )
    parser.add_argument(
        "--data-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAINING")
    )

    # Define command line arguments for hyperparameters with default values directly taken from HyperParams class
    parser.add_argument("--num-epochs", type=int, default=HyperParams.num_epochs)
    parser.add_argument("--batch-size", type=int, default=HyperParams.batch_size)
    parser.add_argument(
        "--learning-rate", type=float, default=HyperParams.learning_rate
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=HyperParams.early_stopping_patience,
    )
    parser.add_argument(
        "--validation-check-frequency",
        type=int,
        default=HyperParams.validation_check_frequency,
    )
    parser.add_argument(
        "--focal-loss-alpha", type=float, default=HyperParams.focal_loss_alpha
    )
    parser.add_argument(
        "--focal-loss-gamma", type=float, default=HyperParams.focal_loss_gamma
    )

    args = parser.parse_args()

    # Create hyperparams instance with values from command line arguments
    hyperparams = HyperParams(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        early_stopping_patience=args.early_stopping_patience,
        validation_check_frequency=args.validation_check_frequency,
        focal_loss_alpha=args.focal_loss_alpha,
        focal_loss_gamma=args.focal_loss_gamma,
    )

    main(args.output_dir, args.data_dir, hyperparams)
