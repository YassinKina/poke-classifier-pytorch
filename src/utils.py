from tqdm import tqdm
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
import math

def get_mean_and_std(dataset):  
    
    """Calculates the mean and std of PokemonDataset
    Args:
        dataset: An instance of the PokemonDataset
    Returns:
        a float tuple[mean, std]
    """
    from src.dataset import PokemonDataset
    
   
    
    # Resize and ToTensor so we can calc the mean and std
    temp_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
    ])

    # Create a temporary dataset instance
    dataset = PokemonDataset(dataset_split=dataset,transform=temp_transform)
    
    # Create a temporary loader without any normalization
    # We use a batch size to speed up the math
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    
    print(f"Calculating stats for {len(dataset)} images...")
    
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _ in tqdm(loader):
        # images shape: [batch_size, 3, height, width]
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        
        # Sum of pixel values per channel
        fst_moment = (fst_moment * cnt + torch.sum(images, dim=[0, 2, 3])) / (cnt + nb_pixels)
        
        # Sum of square of pixel values per channel
        snd_moment = (snd_moment * cnt + torch.sum(images**2, dim=[0, 2, 3])) / (cnt + nb_pixels)
        
        cnt += nb_pixels

    # Standard Deviation formula: sqrt(E[X^2] - (E[X])^2)
    mean = fst_moment
    std = torch.sqrt(snd_moment - fst_moment**2)

    print(f"\nCalculation Complete:")
    print(f"Mean: {mean.tolist()}")
    print(f"Std:  {std.tolist()}")
    
    return mean, std


class NestedProgressBar:
    """A handler for nested tqdm progress bars for training and evaluation loops.

    This class creates and manages an outer progress bar for epochs and an
    inner progress bar for batches. It supports both terminal and Jupyter

    notebook environments and includes a granularity feature to control the
    number of visual updates for very long processes.
    """
    def __init__(
        self,
        total_epochs,
        total_batches,
        g_epochs=None,
        g_batches=None,
        use_notebook=True,
        epoch_message_freq=None,
        batch_message_freq=None,
        mode="train",
    ):
        """Initializes the nested progress bars.

        Args:
            total_epochs (int): The absolute total number of epochs.
            total_batches (int): The absolute total number of batches per epoch.
            g_epochs (int, optional): The visual granularity for the epoch bar.
                                      Defaults to total_epochs.
            g_batches (int, optional): The visual granularity for the batch bar.
                                       Defaults to total_batches.
            use_notebook (bool, optional): If True, uses the notebook-compatible
                                           tqdm implementation. Defaults to True.
            epoch_message_freq (int, optional): Frequency to log epoch
                                                messages. Defaults to None.
            batch_message_freq (int, optional): Frequency to log batch
                                                messages. Defaults to None.
            mode (str, optional): The operational mode, either 'train' or 'eval'.
                                  Defaults to "train".
        """
        self.mode = mode

        # Select the tqdm implementation
        from tqdm.auto import tqdm as tqdm_impl

        self.tqdm_impl = tqdm_impl

        # Store the absolute total counts for epochs and batches
        self.total_epochs_raw = total_epochs
        self.total_batches_raw = total_batches

        # Determine the visual granularity, ensuring it doesn't exceed the total count
        self.g_epochs = min(g_epochs or total_epochs, total_epochs)
        self.g_batches = min(g_batches or total_batches, total_batches)

        # Set the progress bar totals to the calculated granularity
        self.total_epochs = self.g_epochs
        self.total_batches = self.g_batches

        # Initialize the tqdm progress bars based on the operational mode
        if self.mode == "train":
            self.epoch_bar = self.tqdm_impl(
                total=self.total_epochs, desc="Current Epoch", position=0, leave=True
            )
            self.batch_bar = self.tqdm_impl(
                total=self.total_batches, desc="Current Batch", position=1, leave=False
            )
        elif self.mode == "eval":
            self.epoch_bar = None
            self.batch_bar = self.tqdm_impl(
                total=self.total_batches, desc="Evaluating", position=0, leave=False
            )

        # Initialize trackers for the last visualized update step
        self.last_epoch_step = -1
        self.last_batch_step = -1

        # Store the frequency settings for logging messages
        self.epoch_message_freq = epoch_message_freq
        self.batch_message_freq = batch_message_freq

    def update_epoch(self, epoch, postfix_dict=None, message=None):
        """Updates the epoch-level progress bar.

        Args:
            epoch (int): The current epoch number.
            postfix_dict (dict, optional): A dictionary of metrics to display.
                                           Defaults to None.
            message (str, optional): A message to potentially log. Defaults to None.
        """
        # Map the raw epoch count to its corresponding visual step based on granularity
        epoch_step = math.floor((epoch - 1) * self.g_epochs / self.total_epochs_raw)

        # Update the progress bar only when the visual step changes
        if epoch_step != self.last_epoch_step:
            self.epoch_bar.update(1)
            self.last_epoch_step = epoch_step
        # Ensure the progress bar completes on the final epoch
        elif epoch == self.total_epochs_raw and self.epoch_bar.n < self.g_epochs:
            self.epoch_bar.update(1)
            self.last_epoch_step = epoch_step

        # Set the dynamic description for the progress bar
        if self.mode == "train":
            self.epoch_bar.set_description(f"Training - Current Epoch: {epoch}")
        # Update the postfix with any provided metrics or information
        if postfix_dict:
            self.epoch_bar.set_postfix(postfix_dict)

        # Reset the inner batch bar at the start of each new epoch
        self.batch_bar.reset()
        self.last_batch_step = -1

    def update_batch(self, batch, postfix_dict=None, message=None):
        """Updates the batch-level progress bar.

        Args:
            batch (int): The current batch number.
            postfix_dict (dict, optional): A dictionary of metrics to display.
                                           Defaults to None.
            message (str, optional): A message to potentially log. Defaults to None.
        """
        # Map the raw batch count to its corresponding visual step
        batch_step = math.floor((batch - 1) * self.g_batches / self.total_batches_raw)

        # Update the progress bar only when the visual step changes
        if batch_step != self.last_batch_step:
            self.batch_bar.update(1)
            self.last_batch_step = batch_step
        # Ensure the progress bar completes on the final batch
        elif batch == self.total_batches_raw and self.batch_bar.n < self.g_batches:
            self.batch_bar.update(1)
            self.last_batch_step = batch_step

        # Set the dynamic description for the progress bar based on the mode
        if self.mode == "train":
            self.batch_bar.set_description(f"Training - Current Batch: {batch}")
        elif self.mode == "eval":
            self.batch_bar.set_description(f"Evaluation - Current Batch: {batch}")

        # Update the postfix with any provided metrics
        if postfix_dict:
            self.batch_bar.set_postfix(postfix_dict)

    def maybe_log_epoch(self, epoch, message):
        """Logs a message at a specified epoch frequency.

        Args:
            epoch (int): The current epoch number.
            message (str): The message to log.
        """
        if self.epoch_message_freq and epoch % self.epoch_message_freq == 0:
            print(message)

    def maybe_log_batch(self, batch, message):
        """Logs a message at a specified batch frequency.

        Args:
            batch (int): The current batch number.
            message (str): The message to log.
        """
        if self.batch_message_freq and batch % self.batch_message_freq == 0:
            print(message)

    def close(self, last_message=None):
        """Closes all active progress bars and optionally prints a final message.

        Args:
            last_message (str, optional): A final message to print after closing.
                                          Defaults to None.
        """
        # Close the outer epoch bar if it exists (in training mode)
        if self.mode == "train":
            self.epoch_bar.close()
        # Close the inner batch bar
        self.batch_bar.close()

        # Print a concluding message if one is provided
        if last_message:
            print(last_message)