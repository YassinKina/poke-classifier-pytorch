import os
import torch
from datasets import load_dataset, load_from_disk, DatasetDict
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import defaultdict
from imagededup.methods import CNN
from collections import defaultdict
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from src.utils import get_mean_and_std

def download_dataset():
    """Download the pokemon dataaset if it doesn't exist."""

    data_path = "../data"
    os.makedirs(data_path, exist_ok=True)

    # Check if dataset exists locally
    if os.path.exists(os.path.join(data_path, "fcakyon___pokemon-classification")):
        print(f"The pokemon dataset is already downloaded. Loading locally from {data_path}")


        return

    print("Downloading pokemon dataset...")

    # Download dataset from HF
    ds = load_dataset("fcakyon/pokemon-classification", cache_dir=data_path, revision="refs/convert/parquet")

    return

def load_local_data():

    base_path = "../data/fcakyon___pokemon-classification/default/0.0.0/c07895408e16b7f50c16d7fb8abbcae470621248"

    # Point to the specific files because they are all in one directory
    data_files = {
        "train": os.path.join(base_path, "pokemon-classification-train.arrow"),
        "validation": os.path.join(base_path, "pokemon-classification-validation.arrow"),
        "test": os.path.join(base_path, "pokemon-classification-test.arrow")
    }

    # Load via the arrow builder
    ds = load_dataset("arrow", data_files=data_files)
    return ds

def view_dataset(ds, split="train", idx=None):
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

def sanitize_dataset(ds_full):
    """Removes augmented images from raw dataset and returns dataset of all unaugmented images

    Args:
        ds_full (_type_): dataset of raw augmented and orginal images

    Returns:
        _type_: dataset of only unaugmented images
    """
    # Temporary folder to store images for the dedupper to read
    temp_dir = "./temp_images"
    os.makedirs(temp_dir, exist_ok=True)

    print("Exporting images for deep analysis...")
    for idx, example in enumerate(ds_full):
        # Save each image with its index as the name
        example['image'].save(f"{temp_dir}/{idx}.jpg")

    # Use a Pre-trained CNN to find augmentations
    cnn = CNN()

    # This finds all images that are more than 90% similar (covers flips/rotations)
    duplicates = cnn.find_duplicates(image_dir=temp_dir, min_similarity_threshold=0.9)

    # Filter similar, augmented images out
    seen_indices = set()
    unique_indices = []

    for img_name, dupe_list in duplicates.items():
        idx = int(img_name.split('.')[0])
        if idx not in seen_indices:
            unique_indices.append(idx)
            seen_indices.add(idx)
            # Mark all its augmented "siblings" as seen so they are skipped
            for dupe_name in dupe_list:
                dupe_idx = int(dupe_name.split('.')[0])
                seen_indices.add(dupe_idx)

    print(f"Found {len(unique_indices)} unique 'Parent' images out of {len(ds_full)} total.")

    
    return ds_full.select(unique_indices)

def save_dataset_locally(dataset):
    
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

def split_dataset(data_path="./data/pokemon_clean"):
    # Load the master sanitized dataset
    ds = load_from_disk(data_path)
    
    # First split: 80% Train, 20% (Val + Test) with every pokemon in each split
    train_test_split = ds.train_test_split(test_size=0.2, seed=42, stratify_by_column="labels")
    
    # Second split: 10% val, 10% test
    test_val_split = train_test_split['test'].train_test_split(test_size=0.5, seed=42, stratify_by_column="labels")
    
    final_ds = DatasetDict({
        'train': train_test_split['train'],
        'validation': test_val_split['train'], 
        'test': test_val_split['test']        
    })
    
    print(f"Final Counts -> Train: {len(final_ds['train'])}, Val: {len(final_ds['validation'])}, Test: {len(final_ds['test'])}")
    return final_ds

def get_train_test_transforms(mean, std):
        """
        Creates and returns a composition of image transformations for data augmentation
        and preprocessing.

        Args:
            mean (list or tuple): A sequence of mean values for each channel.
            std (list or tuple): A sequence of standard deviation values for each channel.

        Returns:
            torchvision.transforms.Compose: A composed pipeline of train transformations.
            torchvision.transforms.Compose: A composed pipeline of test transformations without augmentations
            
        """
        
        base_transforms = [
            # Resize the input image to 256x256 pixels.
            transforms.Resize((256, 256)),
            # Crop the center 224x224 pixels of the image.
            transforms.CenterCrop(224),
             
        ]
        
        augmentations_transforms = [
        # Randomly flip the image horizontally with a 50% probability.
            transforms.RandomHorizontalFlip(p=0.5),
            # Randomly rotate the image within a range of +/- 10 degrees.
            transforms.RandomRotation(degrees=10),
            # Randomly adjust the brightness of the image.
            transforms.ColorJitter(brightness=0.2),
        ]
        
        main_transforms = [
            # Convert the image to a PyTorch tensor.
            transforms.ToTensor(),
            # Normalize the tensor
            transforms.Normalize(mean=mean, std=std),
        ]
        # base + augmented + main
        transforms_train = transforms.Compose(base_transforms +augmentations_transforms + main_transforms)
        transforms_test = transforms.Compose(base_transforms + main_transforms)
        
        return transforms_train, transforms_test

def create_dataloaders(data_path, batch_size=64, ):
    """ Creates train, val, and test DataLoaders 

    Args:
        data_dir (_type_): _description_
        batch_size (int, optional): _description_. Defaults to 32.

    Returns:
        _type_: _description_
    """
    from src.dataset import PokemonDataset
    # Saved dataset is unsplit, so must split it first
    dataset = split_dataset(data_path)
 
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
    
def verify_dataloaders(train_dl):
    """After running create_dataloaders, check that the dataloader has the correct data

    Args:
        train_dl (_type_): _description_
    """
     # Grab the first batch from the train_loader
    images, labels = next(iter(train_dl))

    print(f"Batch Image Shape: {images.shape}") 
    # Expected: [batch_size, 3, 224, 224] -> e.g., [32, 3, 224, 224]

    print(f"Batch Label Shape: {labels.shape}")
    # Expected: [batch_size] -> e.g., [32]

    print(f"Label Data Type: {labels.dtype}")
    # Expected: torch.int64 (LongTensor)

    print(f"Image Pixel Range: Min={images.min():.2f}, Max={images.max():.2f}")
    # If normalized, you'll see negative numbers and values around 0-2.


def test_main():
    

    download_dataset()
    ds = load_local_data()
    print(ds)

    view_dataset(ds)





