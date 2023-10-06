from regex import D
import torch
from torch.utils.data import DataLoader

from enformer_pytorch import Enformer
from tqdm import tqdm
from multiprocessing import cpu_count

from deepseq.finetune import HeadAdapterWrapper
from deepseq.data import MaskedGenomeIntervalDataset
from deepseq.deepseq import DeepSeq

from transformers import PreTrainedModel


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion,
    device: torch.device,
    scaler=None,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Training"):
        inputs = batch[0].to(device)
        targets = batch[1].to(device)

        optimizer.zero_grad()

        # If scaler is provided, use mixed precision
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(inputs)
            print(targets.shape)
            print(outputs.shape)
            loss = criterion(outputs, targets)

        # If scaler is provided, scale the loss
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    return average_loss


def transfer_enformer_weights_to_(model) -> PreTrainedModel:
    # Load pretrained weights
    enformer = Enformer.from_pretrained("EleutherAI/enformer-official-rough")
    model.transformer.load_state_dict(enformer.transformer.state_dict())
    model.final_pointwise.load_state_dict(enformer.final_pointwise.state_dict())
    return model


def main() -> None:
    # Initialize model and data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    deepseq = DeepSeq.from_hparams(
        dim=1536,
        depth=11,
        heads=8,
        target_length=512,
        return_augs=True,
        rc_aug=True,
        num_downsamples=5,
    ).to(device)

    deepseq = transfer_enformer_weights_to_(deepseq)

    # Freeze transformer weights (we can unfreeze after some training)
    for param in deepseq.transformer.parameters():
        param.requires_grad = False

    model = HeadAdapterWrapper(
        enformer=deepseq,
        num_tracks=2,
    ).to(device)

    # Define dataset and dataloader
    dataset = MaskedGenomeIntervalDataset(
        bed_file="./data/consolidated.bed",
        fasta_file="./data/genome.fa",
        cell_lines_dir="./data/cell_lines/",
        return_augs=False,
        rc_aug=False,
        shift_augs=(-10, 10),
        mask_prob=0.15,
        context_length=16_384,
    )

    num_workers = cpu_count() if torch.cuda.is_available() else 0
    dataloader = DataLoader(
        dataset, batch_size=16, shuffle=True, num_workers=num_workers, pin_memory=True
    )

    # Define loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # TODO
    # scaler = torch.cuda.amp.GradScaler()  # TODO
    # TODO Linear warmup and cosine decay learning rate schedule

    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}")


if __name__ == "__main__":
    main()
