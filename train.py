import torch
import torch.nn as nn
from src.data_setup import create_dataloaders
from src.model import CNN
from src.engine import train_model, init_wandb_run
import os
import random
import wandb


def main():
    torch.manual_seed(42) 
    
    # 1. Configuration & Hyperparameters
    DATA_DIR = "./data/pokemon_clean"
    BATCH_SIZE = 64
    LEARNING_RATE = 0.0003
    NUM_EPOCHS = 25
    NUM_CLASSES = 150  
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    ARCHITECTURE = "CNN"
    DATASET_NAME = "fcakyon/pokemon-classification"
    
    # Model architecture parameters
    N_LAYERS = 4
    N_FILTERS = [32, 64, 128, 256]
    KERNEL_SIZES = [3, 3, 3, 3]
    DROPOUT_RATE = 0.2
    FC_SIZE = 512
    
   # wandb config
    config = {
        "learning_rate": LEARNING_RATE,
        "architecture": ARCHITECTURE,
        "dataset": DATASET_NAME,
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "dropout": DROPOUT_RATE,
        "n_layers": N_LAYERS
    }
    
    wandb_run = init_wandb_run(config=config)

    print(f"--- Training on Device: {DEVICE} ---")

    # 2. Setup DataLoaders
    train_dl, val_dl, test_dl = create_dataloaders(data_path=DATA_DIR, batch_size=BATCH_SIZE)

    # 3. Initialize Model
    model = CNN(
        n_layers=N_LAYERS,
        n_filters=N_FILTERS,
        kernel_sizes=KERNEL_SIZES,
        dropout_rate=DROPOUT_RATE,
        fc_size=FC_SIZE,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    # 4. Define Loss and Optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    

    # 5. Start Training
    wandb_run = init_wandb_run(config=config)
    
    train_model(
        model=model,
        optimizer=optimizer,
        loss_func=loss_func,
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        device=DEVICE,
        n_epochs=NUM_EPOCHS,
        wandb_run=wandb_run
    )

  

if __name__ == "__main__":
    main()