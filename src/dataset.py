from torch.utils.data import Dataset
import torch
import numpy as np
from torchvision import transforms
from datasets import load_from_disk
from PIL import Image
from typing import Any, Tuple, Optional, List, Dict
import os

class PokemonDataset(Dataset):
    """
    A custom PyTorch Dataset for Pokémon image classification using Hugging Face datasets.

    This class interfaces with a Hugging Face dataset split to provide transformed 
    images and their corresponding labels for model training and evaluation.
    """

    def __init__(self, dataset_split: Dataset, transform: Optional[transforms.Compose] = None, image_dir: Optional[str] ="data/temp_images_global"):
        """
        Initializes the PokemonDataset with a specific dataset split and optional transforms.

        Args:
            dataset_split (Any): A Hugging Face Dataset split containing 'image' and 'labels'.
            transform (Optional[transforms.Compose]): PyTorch transforms for data 
                augmentation and normalization.
            
        """
        self.image_dir = image_dir
        self.transform = transform
        self.dataset = dataset_split
        self.labels = self.load_labels(self.dataset)

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The number of images in the dataset split.
        """
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Retrieves a processed image and its label for a given index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            Tuple[torch.Tensor, int]: A tuple containing the transformed image tensor 
                and the integer class label.
        """
        
        # Loop to attempt loading a valid sample, preventing an infinite loop.
        for attempt in range(len(self)):
            # Attempt to load and process the sample.
            try:
                # Retrieve the image using the helper method.
                item = self.retrieve_item(idx)
                
                image = item['image']
                label = item['labels']
                
                #Ensure image is RGB
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                    
                # Check if a transform has been provided.
                if self.transform:
                    # Apply the transform to the image.
                    image = self.transform(image)
                
                # Return the valid image and its corresponding label.
                return image, label
            # Catch any exception that occurs during the process.
            except Exception as e:
                # Log the error with its index and message.
                self.log_error(idx, e)
                # Move to the next index, wrapping around if necessary.
                idx = (idx + 1) % len(self)
    
    def retrieve_item(self, idx: int) -> Dict[str, Any]:
        """
        Accesses a single raw entry from the underlying dataset split.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            Dict[str, Any]: A dictionary containing the raw PIL Image and label data.
        """
        return self.dataset[idx]
        
    def load_labels(self, dataset: Any) -> List[int]:
        """
        Extracts the full list of labels from the dataset split.

        Args:
            dataset (Any): The dataset split to extract labels from.

        Returns:
            List[int]: A list of integer labels corresponding to the dataset samples.
        """
        labels = dataset["labels"]
        return labels
    
    def log_error(self, idx, e):
        """
        Records the details of an error encountered during data loading.

        Args:
            idx (int): The index of the problematic sample.
            e (Exception): The exception object that was raised.
        """
        # Construct the filename of the problematic image.
        img_name = f"{idx}.jpg"
        # Construct the full path to the image file.
        img_path = os.path.join(self.img_dir, img_name)
        # Append a dictionary with error details to the log.
        self.error_logs.append(
            {
                "index": idx,
                "error": str(e),
                "path": img_path if "img_path" in locals() else "unknown",
            }
        )
        # Print a warning to the console about the skipped image.
        print(f"Warning: Skipping corrupted image {idx}: {e}")
        
    def get_error_summary(self):
        """
        Prints a summary of all errors encountered during dataset processing.
        """
        # Check if the error log is empty.
        if not self.error_logs:
            # Print a message indicating the dataset is clean.
            print("No errors encountered - dataset is clean!")
        else:
            # Print the total number of problematic images found.
            print(f"\nEncountered {len(self.error_logs)} problematic images:")
            # Iterate through the first few logged errors.
            for error in self.error_logs[:5]:
                # Print the details of an individual error.
                print(f"  Index {error['index']}: {error['error']}")
            # Check if there are more errors than were displayed.
            if len(self.error_logs) > 5:
                # Print a summary of the remaining errors.
                print(f"  ... and {len(self.error_logs) - 5} more")