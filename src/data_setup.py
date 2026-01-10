import os
import torch
from datasets import load_dataset, load_from_disk, DatasetDict, concatenate_datasets
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import defaultdict
from imagededup.methods import CNN
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from src.utils import get_mean_and_std
import glob
from typing import Optional, Tuple, List, Union

def download_dataset(raw_path: str) -> None:
    """
    Downloads the Pokémon dataset from Hugging Face if it does not already exist.

    Args:
        raw_path (str): The local directory path where the dataset should be stored.
    """
    os.makedirs(raw_path, exist_ok=True)
    print("Downloading pokemon dataset...")
    # Download dataset from HF
    ds = load_dataset("fcakyon/pokemon-classification", cache_dir=raw_path, revision="refs/convert/parquet")
    return

def load_local_data() -> DatasetDict:
    """
    Locates and loads local .arrow files into a Hugging Face DatasetDict.

    Returns:
        DatasetDict: The loaded dataset containing train, validation, and test splits.
    
    Raises:
        FileNotFoundError: If no .arrow files are found in the expected directory.
    """
    # Use the root of your data directory
    base_search_path = "./data/fcakyon___pokemon-classification"
    
    # Look for any .arrow files recursively within that folder
    arrow_files = glob.glob(os.path.join(base_search_path, "**/*.arrow"), recursive=True)
    
    # Map .arrow files into splits
    data_files = {}
    for f in arrow_files:
        if "train" in f: data_files["train"] = f
        elif "validation" in f or "valid" in f: data_files["validation"] = f
        elif "test" in f: data_files["test"] = f

    if not data_files:
        raise FileNotFoundError(f"No .arrow files found in {base_search_path}. Check if the download actually completed.")

    print(f"Found local data files: {list(data_files.keys())}")
    return load_dataset("arrow", data_files=data_files)

def view_dataset(ds: DatasetDict, split: str = "train", idx: Optional[int] = None) -> None:
    """
    Visualizes a single sample from the dataset with its associated Pokémon name and ID.

    Args:
        ds (DatasetDict): The dataset to visualize.
        split (str): The dataset split to sample from (e.g., 'train').
        idx (Optional[int]): Specific index to view; if None, a random index is chosen.
    """
    # Get the total number of images in the split
    dataset_size = len(ds[split])

    # Pick a random index
    random_idx = random.randint(0, dataset_size - 1)
    
    if idx is not None and idx >= 0 and idx < dataset_size - 1:
        random_idx = idx

    # Grab the random sample
    sample = ds[split][random_idx]
    img = sample['image']
    label_id = sample['labels']

    # Extract class names from metadata
    class_names = ds[split].features['labels'].names
    pokemon_name = class_names[label_id]

    # Display pokemon
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f"Name: {pokemon_name}\nID: {label_id} (Index: {random_idx})")
    plt.axis('off')
    plt.show()

def sanitize_dataset(save_path: str) -> DatasetDict:
    """
    Performs global deduplication across all splits and re-stratifies the data into 80/10/10 splits.

    Args:
        save_path (str): The directory path to save the cleaned DatasetDict.

    Returns:
        DatasetDict: The deduplicated and re-split dataset.
    """
    #  Load the existing arrow files (which currently have duplicates)
    ds_full_dict = load_local_data()
    
    # Flatten into one single Dataset
    print("Combining all splits for global deduplication...")
    combined_ds = concatenate_datasets([
        ds_full_dict["train"], 
        ds_full_dict["validation"], 
        ds_full_dict["test"]
    ])
    
    # Export images to one temp folder for the CNN
    temp_dir = "./data/temp_images_global"
    os.makedirs(temp_dir, exist_ok=True)
    
    for idx, example in enumerate(combined_ds):
        example['image'].save(f"{temp_dir}/{idx}.jpg")

    # Use CNN to find duplicates globally
    cnn = CNN() 
    duplicates = cnn.find_duplicates(image_dir=temp_dir, min_similarity_threshold=0.9)

    # Identify unique indices
    seen_indices = set()
    unique_indices = []
    for img_name, dupe_list in sorted(duplicates.items(), key=lambda x: int(x[0].split('.')[0])):
        idx = int(img_name.split('.')[0])
        if idx not in seen_indices:
            unique_indices.append(idx)
            seen_indices.add(idx)
            for dupe_name in dupe_list:
                seen_indices.add(int(dupe_name.split('.')[0]))

    unique_ds = combined_ds.select(unique_indices)
    print(f"Global deduplication complete: {len(unique_ds)} unique images remaining.")

    #  Re-split into 80/10/10 (Stratified)
    train_temp = unique_ds.train_test_split(test_size=0.2, seed=42, stratify_by_column="labels")
    val_test = train_temp["test"].train_test_split(test_size=0.5, seed=42, stratify_by_column="labels")

    final_ds = DatasetDict({
        "train": train_temp["train"],
        "validation": val_test["train"],
        "test": val_test["test"]
    })
    # Saved newly split dataset locally
    final_ds.save_to_disk(save_path)
    print(f"Cleaned and re-stratified dataset saved at {save_path}")
    return final_ds

def save_dataset_locally(dataset: DatasetDict) -> None:
    """
    Saves a DatasetDict to a project-relative 'data/pokemon_clean' directory.

    Args:
        dataset (DatasetDict): The dataset object to be saved.
    """
    # Get the path of the current file (data_setup.py)
    current_file = Path(__file__).resolve()

    # Go up one level to the project root
    project_root = current_file.parent.parent

    # Define the data folder relative to the root
    folder_path = project_root / "data" / "pokemon_clean"

    # Ensure the directory exists
    os.makedirs(folder_path, exist_ok=True)
    
    # Save the dataset
    print(f"Saving dataset to {folder_path}...")
    dataset.save_to_disk(folder_path)
    print("Save complete!")

def split_dataset(cleaned_path: str = "data/pokemon_clean") -> DatasetDict:
    """
    Loads a cleaned dataset if available; otherwise, handles downloading and sanitization.

    Args:
        cleaned_path (str): Path to the saved cleaned dataset.

    Returns:
        DatasetDict: The master sanitized dataset containing all splits.
    """
    ds =  None
    raw_path = "data/fcakyon___pokemon-classification"
    # Create cleaned dataset with balanced labels representation
    if not os.path.exists(cleaned_path):
        if not os.path.exists(raw_path):
            print("Raw dataset not found. Downloading data")
            download_dataset(raw_path)
            
        print("Clean dataset not found. Running sanitization script...")
        ds = sanitize_dataset(save_path="data/pokemon_clean")
    else:
        # Load the master sanitized dataset
        print(f"Loading cleaned data from {cleaned_path}")
        ds = load_from_disk(cleaned_path)
        
    # If we already have a DatasetDict, return
    print(f"Final Counts -> Train: {len(ds['train'])}, Val: {len(ds['validation'])}, Test: {len(ds['test'])}")
    return ds
    
def get_train_test_transforms(mean: Union[List[float], Tuple[float, ...]], std: Union[List[float], Tuple[float, ...]]) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Defines the transformation pipelines for training (with augmentation) and testing.

    Args:
        mean (Union[List[float], Tuple[float, ...]]): Normalization means for RGB channels.
        std (Union[List[float], Tuple[float, ...]]): Normalization standard deviations for RGB channels.

    Returns:
        Tuple[transforms.Compose, transforms.Compose]: A tuple containing (train_transforms, test_transforms).
    """
    base_transforms = [
        # Resize the input image to 256x256 pixels.
        transforms.Resize((256, 256)),
        # Crop the center 224x224 pixels of the image.
        transforms.CenterCrop(224),
    ]
    
    augmentations_transforms = [
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        # Add contrast and saturation to ColorJitter
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        # Randomly turn the image grayscale (5% of the time)
        transforms.RandomGrayscale(p=0.05),
    ]
    
    main_transforms = [
        # Convert the image to a PyTorch tensor.
        transforms.ToTensor(),
        # Normalize the tensor
        transforms.Normalize(mean=mean, std=std),
    ]
    # base + augmented + main
    transforms_train = transforms.Compose(base_transforms + augmentations_transforms + main_transforms)
    transforms_test = transforms.Compose(base_transforms + main_transforms)
    
    return transforms_train, transforms_test

def create_dataloaders(clean_data_path: str, batch_size: int = 64) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Initializes PyTorch DataLoaders for the train, validation, and test splits.

    Args:
        clean_data_path (str): Local path to the sanitized dataset.
        batch_size (int): Number of samples per batch.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: (train_loader, val_loader, test_loader).
    """
    from src.dataset import PokemonDataset
    # Saved dataset is unsplit, so must split it first
    dataset = split_dataset(clean_data_path)
 
    # Calculate the stats using this un-normalized data
    train_mean, train_std = get_mean_and_std(dataset=dataset["train"])
    
    #Get train and test transforms
    train_transforms, test_transforms = get_train_test_transforms(mean=train_mean, std=train_std)
    
    # Setup the three distinct datasets
    train_data = PokemonDataset(dataset_split=dataset["train"], transform=train_transforms)
    val_data   = PokemonDataset(dataset_split=dataset["validation"], transform=test_transforms)
    test_data  = PokemonDataset(dataset_split=dataset["test"], transform=test_transforms)

    # Create the three loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
    
def verify_dataloaders(train_dl: DataLoader) -> None:
    """
    Prints diagnostic information about the first batch of a DataLoader to verify integrity.

    Args:
        train_dl (DataLoader): The DataLoader to inspect.
    """
    # Grab the first batch from the train_loader
    images, labels = next(iter(train_dl))

    print(f"Batch Image Shape: {images.shape}") 
    print(f"Batch Label Shape: {labels.shape}")
    print(f"Label Data Type: {labels.dtype}")
    print(f"Image Pixel Range: Min={images.min():.2f}, Max={images.max():.2f}")

def test_main() -> None:
    """
    Orchestrates a basic end-to-end test of the data downloading and loading pipeline.
    """
    download_dataset("data/fcakyon___pokemon-classification")
    ds = load_local_data()
    print(ds)
    view_dataset(ds)