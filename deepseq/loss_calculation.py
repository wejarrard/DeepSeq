# loss_calculation.py
import torch
from torch.utils.tensorboard import SummaryWriter


class TrainLossTracker:
    def __init__(
        self,
        criterion,
        device,
        writer: SummaryWriter,
        check_frequency=1,
    ):
        self.criterion = criterion
        self.device = device
        self.writer = writer
        self.check_frequency = check_frequency
        self.frequency_counter = 0
        self.loss_accumulator = 0.0  # Accumulates the loss values
        self.batch_counter = 0  # Counts the actual number of loss values added
        self.global_step = 0

    def __call__(self, loss):
        # Accumulate loss and increment batch counter irrespective of the logging frequency
        self.loss_accumulator += loss
        self.batch_counter += 1

        # Only log the average loss at the specified frequency
        self.frequency_counter += 1
        if self.frequency_counter < self.check_frequency:
            return None  # Logging not yet performed

        # Reset the counter for frequency checking
        self.frequency_counter = 0

        # Calculate the average loss
        if self.batch_counter == 0:
            avg_loss = 0
        else:
            avg_loss = self.loss_accumulator / self.batch_counter

        # Log the average training loss to TensorBoard
        self.writer.add_scalar("Loss/train", avg_loss, self.global_step)

        # Increment the global step for TensorBoard
        self.global_step += 1

        # Reset the loss accumulator and batch counter after logging
        self.loss_accumulator = 0.0
        self.batch_counter = 0

        return avg_loss


class ValidationLossCalculator:
    def __init__(
        self,
        val_dataloader,
        criterion,
        device,
        writer: SummaryWriter,
        check_frequency=1,
    ):
        self.val_dataloader = val_dataloader
        self.criterion = criterion
        self.device = device
        self.writer = writer
        self.check_frequency = check_frequency
        self.frequency_counter = 0
        self.global_step = 0

    def __call__(self, model):
        self.frequency_counter += 1
        if self.frequency_counter < self.check_frequency:
            return None, None

        self.frequency_counter = 0
        model.eval()

        total_val_loss = 0.0
        correct_predictions = 0  # track the number of correct predictions
        total_predictions = 0  # track the total number of predictions

        with torch.no_grad():
            for batch in self.val_dataloader:
                inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                outputs = model(inputs)
                loss = self.criterion(outputs, targets)
                total_val_loss += loss.item()

                # Calculating accuracy
                predicted = (
                    torch.sigmoid(outputs) >= 0.5
                )  # Convert outputs to probabilities and then to 0 or 1
                correct_predictions += (
                    (predicted == targets).sum().item()
                )  # Sum all correct predictions in the batch
                total_predictions += (
                    targets.numel()
                )  # Increment total predictions by the number of elements in targets

        average_val_loss = total_val_loss / len(self.val_dataloader)
        val_accuracy = correct_predictions / total_predictions

        # Log values to TensorBoard
        self.writer.add_scalar("Loss/val", average_val_loss, self.global_step)
        self.writer.add_scalar("Accuracy/val", val_accuracy, self.global_step)

        self.global_step += 1

        model.train()

        return average_val_loss, val_accuracy
