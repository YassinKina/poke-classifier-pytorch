from tqdm import tqdm
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
import math
import numpy as np
import random
import os
import wandb
from typing import Tuple, Any, Optional
from omegaconf import DictConfig, OmegaConf
from torchview import draw_graph
from src import DynamicCNN

POKEMON_MEAN = torch.tensor([0.5863186717033386, 0.5674829483032227, 0.5336665511131287])
POKEMON_STD = torch.tensor([0.34640103578567505, 0.33123084902763367, 0.34212544560432434])

def get_mean_and_std(dataset: Optional[Any]= None, fast: bool= True) -> Tuple[torch.Tensor, torch.Tensor]:  
    """
    Calculates the channel-wise mean and standard deviation for the dataset.

    This function iterates through the dataset using a temporary DataLoader to compute 
    the first and second raw moments of the pixel intensities. These moments are 
    then used to derive the global mean and standard deviation, which are essential 
    for proper image normalization during training and inference.
    
    If fast is True, we skip calculations and return the previously calculated mean and std

    Args:
        dataset (Any): A Hugging Face Dataset split (e.g., ds['train']) or an 
            un-normalized dataset object.
        fast (bool): A switch that determines whether the mean and std is recalculated

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the mean and 
            standard deviation tensors, each with shape [3] (one per RGB channel).
    """
    # Avoid calcing same mean and std every time we train/eval
    if fast:
       return POKEMON_MEAN, POKEMON_STD
   

    from .dataset import PokemonDataset
        
    # Resize and ToTensor so we can calc the mean and std
    temp_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Create a temporary dataset instance
    dataset_obj = PokemonDataset(dataset_split=dataset, transform=temp_transform)
    
    # Create a temporary loader without any normalization
    # We use a batch size to speed up the math
    loader = DataLoader(dataset_obj, batch_size=64, shuffle=False, num_workers=0)
    
    print(f"Calculating stats for {len(dataset_obj)} images...")
    
    cnt = 0
    fst_moment = torch.zeros(3)
    snd_moment = torch.zeros(3)

    for images, _ in tqdm(loader):
        # images shape: [batch_size, 3, height, width]
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        
        # Sum of pixel values per channel
        # Calculate moving averages of moments to prevent overflow on large datasets
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

def get_best_val_accuracy(model_path: str = "models/pokemon_cnn_best.pth") -> float:
    """
    Loads the best saved model checkpoint and retrieves the recorded accuracy.
    
    Args:
        model_path (str): Path to the saved .pth checkpoint.
        
    Returns:
        float: The best validation accuracy found, or 0.0 if no model exists.
    """
    if os.path.exists(model_path):
        try:
            # map_location ensures it loads even if saved on a different device
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            
            # If you saved it as a dict, return the 'accuracy' key
            if isinstance(checkpoint, dict) and 'accuracy' in checkpoint:
                return checkpoint['accuracy']
            
            # Fallback for old models that only saved the state_dict
            return 0.0 
        except Exception as e:
            print(f"Warning: Could not read accuracy from checkpoint: {e}")
            return 0.0
    return 0.0

def flatten_config(raw_dict, parent_key='', sep='/'):
    flat_config = dict()
    
    for key, value in raw_dict.items():
        if isinstance(value, dict):
            # If it's a group (like 'training'), pull everything out
            for sub_key, sub_value in value.items():
                flat_config[sub_key] = sub_value
        else:
            # If it's a top-level setting (like 'device' or 'project_name')
            flat_config[key] = value
            
    return flat_config

def set_seed(seed=42):
    """Set constant random see for all training

    Args:
        seed (int, optional): Random seed that fefaults to 42.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # safe to call even on Mac
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
def get_num_correct_in_top5(outputs: torch.Tensor, labels: torch.Tensor) -> int:
    """
    Calculates the number of targets present within the top 5 model predictions.

    This function determines 'Top-5 accuracy' counts by identifying the indices 
    of the 5 highest logits for each sample and checking if the ground truth 
    label exists among them. T

    Args:
        outputs (torch.Tensor): Raw model logits of shape (batch_size, num_classes).
        labels (torch.Tensor): Ground truth class indices of shape (batch_size).

    Returns:
        int: Total number of samples in the current batch where the target label 
            was within the top 5 predicted indices.
    """
    # Expand labels from [batch_size] to [batch_size, 1] to compare against [batch_size, 5]
    _, top5_indices = outputs.topk(5, 1, largest=True, sorted=True)
    labels_reshaped = labels.view(-1, 1).expand_as(top5_indices)
    
    # See if the correct labels occurs in the top 5 predictions
    correct_in_top5 = (top5_indices == labels_reshaped).any(dim=1).sum().item()
    
    return correct_in_top5

def init_wandb_run(config: DictConfig, run_name: str) -> Any:
    """
    Initializes a Weights & Biases run with flattened configuration parameters.

    Converts Hydra DictConfigs to standard dictionaries and prepares the environment
    for a stable W&B initialization, specifically optimized for macOS threading.

    Args:
        config (DictConfig): The Hydra configuration containing hyperparameters.
        run_name (str): The descriptive name for the W&B run.

    Returns:
        Any: The initialized W&B run object.
    """
    os.environ["WANDB_START_METHOD"] = "thread"
    config_dict = OmegaConf.to_container(config, resolve=True)
    flat_config = flatten_config(config_dict)

    run = wandb.init(
        entity="yassinbkina",
        project="pokemon-classification",
        config=flat_config, 
        name=run_name,
        group="GAP_hpo",
        settings=wandb.Settings(start_method="thread"), 
        reinit=True
    )
    
    print("WANDB config: ", wandb.config)
    return run
    
def get_list_labels():
    labels = ["Golbat", "Machoke", "Omastar", "Diglett", "Lapras", "Kabuto", "Persian", "Weepinbell", "Golem", "Dodrio", "Raichu", "Zapdos", "Raticate", "Magnemite", 
    "Ivysaur", "Growlithe", "Tangela", "Drowzee", "Rapidash", "Venonat", "Pidgeot", "Nidorino", "Porygon", "Lickitung", "Rattata", "Machop", "Charmeleon", "Slowbro", "Parasect", 
    "Eevee", "Starmie", "Staryu", "Psyduck", "Dragonair", "Magikarp", "Vileplume", "Marowak", "Pidgeotto", "Shellder", "Mewtwo", "Farfetchd", "Kingler", "Seel", "Kakuna", "Doduo", 
    "Electabuzz", "Charmander", "Rhyhorn", "Tauros", "Dugtrio", "Poliwrath", "Gengar", "Exeggutor", "Dewgong", "Jigglypuff", "Geodude", "Kadabra", "Nidorina", "Sandshrew", "Grimer", 
    "MrMime", "Pidgey", "Koffing", "Ekans", "Alolan Sandslash", "Venusaur", "Snorlax", "Paras", "Jynx", "Chansey", "Hitmonchan", "Gastly", "Kangaskhan", "Oddish", "Wigglytuff", "Graveler", 
    "Arcanine", "Clefairy", "Articuno", "Poliwag", "Abra", "Squirtle", "Voltorb", "Ponyta", "Moltres", "Nidoqueen", "Magmar", "Onix", "Vulpix", "Butterfree", "Krabby", "Arbok", "Clefable",
    "Goldeen", "Magneton", "Dratini", "Caterpie", "Jolteon", "Nidoking", "Alakazam", "Dragonite", "Fearow", "Slowpoke", "Weezing", "Beedrill", "Weedle", "Cloyster", "Vaporeon", "Gyarados", 
    "Golduck", "Machamp", "Hitmonlee", "Primeape", "Cubone", "Sandslash", "Scyther", "Haunter", "Metapod", "Tentacruel", "Aerodactyl", "Kabutops", "Ninetales", "Zubat", "Rhydon", "Mew", "Pinsir",
    "Ditto", "Victreebel", "Omanyte", "Horsea", "Pikachu", "Blastoise", "Venomoth", "Charizard", "Seadra", "Muk", "Spearow", "Bulbasaur", "Bellsprout", "Electrode", "Gloom", "Poliwhirl", "Flareon",
    "Seaking", "Hypno", "Wartortle", "Mankey", "Tentacool", "Exeggcute", "Meowth"]
    return labels


    
    
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