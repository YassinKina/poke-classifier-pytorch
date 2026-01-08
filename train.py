import torch
import torch.nn as nn
from src.data_setup import create_dataloaders
from src.model import CNN
from src.engine import train_model
import os

def main():
    # 1. Configuration & Hyperparameters
    DATA_DIR = "./data/pokemon_clean"
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 10
    NUM_CLASSES = 150  
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Model architecture parameters
    N_LAYERS = 4
    N_FILTERS = [32, 64, 128, 256]
    KERNEL_SIZES = [3, 3, 3, 3]
    DROPOUT_RATE = 0.5
    FC_SIZE = 512

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
    train_model(
        model=model,
        optimizer=optimizer,
        loss_func=loss_func,
        train_dataloader=train_dl,
        device=DEVICE,
        n_epochs=NUM_EPOCHS
    )

    #Create the directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Save the state_dict 
    save_path = os.path.join("models", "pokemon_cnn_base.pth")
    torch.save(model.state_dict(), save_path)

    print(f"Model weights saved to {save_path}")

if __name__ == "__main__":
    main()