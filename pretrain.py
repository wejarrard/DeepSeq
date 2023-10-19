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

from deepseq.data import MaskedGenomeIntervalDataset
from deepseq.deepseq import DeepSeq
from deepseq.earlystopping import EarlyStopping
from deepseq.finetune import HeadAdapterWrapper
from deepseq.training_utils import (
    TrainingParams,
    WarmupCosineSchedule,
    count_directories,
    train_one_epoch,
    transfer_enformer_weights_to_,
)

# hide user warning
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class HyperParams:
    num_epochs: int = 2
    batch_size: int = 32 * torch.cuda.device_count() if torch.cuda.is_available() else 4
    learning_rate: float = 1e-4
    early_stopping_patience: int = 20
    early_stopping_check_frequency: int = 10_000


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
    valid_size = 20000
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

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hyperparams.learning_rate,
        weight_decay=0.1,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scaler = GradScaler()

    total_steps = len(train_loader) * hyperparams.num_epochs
    warmup_steps = 0.1 * total_steps

    scheduler = WarmupCosineSchedule(
        optimizer, warmup_steps=warmup_steps, total_steps=total_steps
    )

    training_params = TrainingParams(
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scaler=scaler,
        scheduler=scheduler,
    )

    early_stopping = EarlyStopping(
        model=model,
        val_dataloader=valid_loader,
        model_params=training_params,
        patience=hyperparams.early_stopping_patience,
        check_frequency=hyperparams.early_stopping_check_frequency,  # patience * check_frequency = number of batches without improvement
        verbose=True,
        save_path=os.path.join(output_dir, "best_model.pth"),
    )

    ############ TRAINING ############

    for epoch in range(hyperparams.num_epochs):
        train_loss = train_one_epoch(
            model=model,
            params=training_params,
            train_loader=train_loader,
            early_stopping=early_stopping,
        )

        print(f"Epoch {epoch} train loss: {train_loss:.6f}")
        print("-" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DeepSeq model on SageMaker.")
    parser.add_argument(
        "--output-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", None)
    )
    parser.add_argument(
        "--data-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAINING", None)
    )
    args = parser.parse_args()

    hyperparams = HyperParams()

    main(args.output_dir, args.data_dir, hyperparams)
