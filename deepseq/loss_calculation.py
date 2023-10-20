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
        self.writer = writer  # TensorBoard SummaryWriter
        self.check_frequency = check_frequency
        self.frequency_counter = 0
        self.global_step = 0  # you can also pass this as an argument if it's being tracked outside this class

    def __call__(self, model):
        # Only calculate the loss at the specified frequency
        self.frequency_counter += 1
        if self.frequency_counter < self.check_frequency:
            return None  # or you could return a default value indicating no calculation was done

        # Reset the counter and proceed with the calculation
        self.frequency_counter = 0

        model.eval()  # Set the model to evaluation mode
        total_val_loss = 0.0

        with torch.no_grad():  # Disable gradient calculation during validation
            for batch in self.val_dataloader:
                inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                outputs = model(inputs)
                loss = self.criterion(outputs, targets)
                print(loss.item())
                total_val_loss += loss.item()

        average_val_loss = total_val_loss / len(self.val_dataloader)

        # Log the average validation loss to TensorBoard
        self.writer.add_scalar("Loss/val", average_val_loss, self.global_step)
        self.global_step += 1  # increment step

        model.train()  # Set the model back to training mode

        return average_val_loss
